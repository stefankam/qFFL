import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
import os
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad, norm_grad, norm_grad_sparse

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        self.client_model = None  # ensure attribute exists even if init fails
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.q, self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.data_partition_seed, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.client_id_to_index = {client.id: idx for idx, client in enumerate(self.clients)}
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        self.start_round = 0
        if self.resume_from:
            self.start_round = self.restore() + 1
            print('Resuming from checkpoint, starting at round {}'.format(self.start_round))

    def __del__(self):
        cm = getattr(self, "client_model", None)
        if cm is not None:
            cm.close()

    def setup_clients(self, dataset, data_partition_seed, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u],  data_partition_seed, model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self):
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.train_error() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        return num_samples, tot_correct


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        return num_samples, tot_correct

    def fairness_stats(self, num_correct, num_samples):
        accuracies = np.divide(np.asarray(num_correct), np.asarray(num_samples))
        num_users = len(accuracies)
        if num_users == 0:
            return 0.0, 0.0, 0.0
        num_tail = max(1, int(num_users * 0.1))
        sorted_acc = np.sort(accuracies)
        worst = float(np.mean(sorted_acc[:num_tail]))
        best = float(np.mean(sorted_acc[-num_tail:]))
        variance = float(np.var(accuracies))
        return worst, best, variance

    def validate(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.validate()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        return num_samples, tot_correct

    def test_resulting_model(self):
        num_samples = []
        tot_correct = []
        #self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass



    def _init_metrics_storage(self):
        total_rounds = self.num_rounds + 1
        self.metrics_rounds = total_rounds
        self.metrics_availability = np.ones((total_rounds, len(self.clients)), dtype=int)
        self.metrics_selected = [[] for _ in range(total_rounds)]
        self.metrics_loss_before = np.full((total_rounds, len(self.clients)), np.nan)
        self.metrics_loss_after = np.full((total_rounds, len(self.clients)), np.nan)
        self.metrics_accuracy = np.full((total_rounds, len(self.clients)), np.nan)

    def record_round_metrics(self, round_num, selected_ids, loss_before, loss_after):
        if not self.metrics_npz:
            return
        if not hasattr(self, "metrics_selected"):
            self._init_metrics_storage()
        self.metrics_selected[round_num] = list(selected_ids)
        for client_id, before, after in zip(selected_ids, loss_before, loss_after):
            client_index = self.client_id_to_index.get(client_id)
            if client_index is None:
                continue
            self.metrics_loss_before[round_num, client_index] = before
            self.metrics_loss_after[round_num, client_index] = after

    def record_round_accuracy(self, round_num, accuracies):
        if not self.metrics_npz:
            return
        if not hasattr(self, "metrics_selected"):
            self._init_metrics_storage()
        self.metrics_accuracy[round_num, :] = np.asarray(accuracies, dtype=float)

    def save_metrics_npz(self, round_num=None):
        if not self.metrics_npz:
            return
        if not hasattr(self, "metrics_selected"):
            self._init_metrics_storage()
        output_path = self.metrics_npz
        if round_num is not None:
            base, ext = os.path.splitext(output_path)
            if not ext:
                ext = ".npz"
            output_path = f"{base}_round{round_num}{ext}"
        np.savez(
            output_path,
            rounds=self.metrics_rounds,
            num_clients=len(self.clients),
            m_per_round=self.clients_per_round,
            availability=self.metrics_availability,
            selected=np.array(self.metrics_selected, dtype=object),
            loss_before=self.metrics_loss_before,
            loss_after=self.metrics_loss_after,
            accuracy=self.metrics_accuracy,
            accuracy_final=self.metrics_accuracy[-1],
        )
    def _checkpoint_prefix(self):
        if not self.checkpoint_dir:
            return None
        return os.path.join(self.checkpoint_dir, 'model.ckpt')

    def save_checkpoint(self, round_num):
        checkpoint_prefix = self._checkpoint_prefix()
        if not checkpoint_prefix:
            return
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.client_model.saver.save(self.client_model.sess, checkpoint_prefix, global_step=round_num)

    def restore(self):
        resume_path = self.resume_from
        if not resume_path:
            return 0
        if os.path.isdir(resume_path):
            resume_path = tf.train.latest_checkpoint(resume_path)
        if not resume_path:
            print('Warning: no checkpoint found to resume from. Starting from round 0.')
            return 0
        self.client_model.saver.restore(self.client_model.sess, resume_path)
        self.latest_model = self.client_model.get_params()
        return self._extract_round(resume_path)

    def _extract_round(self, checkpoint_path):
        basename = os.path.basename(checkpoint_path)
        if '-' not in basename:
            return 0
        try:
            return int(basename.rsplit('-', 1)[-1])
        except ValueError:
            return 0

    def _read_cpu_times(self):
        try:
            with open('/proc/stat', 'r') as stats:
                cpu_line = stats.readline()
        except OSError:
            return None
        parts = cpu_line.split()
        if len(parts) < 5:
            return None
        times = [int(v) for v in parts[1:]]
        total = sum(times)
        idle = times[3]
        return total, idle

    def _cpu_usage_percent(self, previous, current):
        if not previous or not current:
            return 0.0
        total_delta = current[0] - previous[0]
        idle_delta = current[1] - previous[1]
        if total_delta <= 0:
            return 0.0
        return (1.0 - (idle_delta / total_delta)) * 100.0



    def select_clients(self, round, pk, held_out=None, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            indices: an array of indices
            self.clients[]
        '''
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round+4)

        
        if held_out: # meta-learning
            can_be_chosen = len(self.clients) - held_out
        else:
            can_be_chosen = len(self.clients)


        if self.sampling == 1:  # uniform sampling + weighted average
            indices = np.random.choice(range(can_be_chosen), num_clients, replace=False)
            return indices, np.asarray(self.clients)[indices]
                   
        elif self.sampling == 2:  # weighted sampling + simple average
            num_samples = []
            for client in self.clients[:can_be_chosen]:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(can_be_chosen), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

        else:
            raise ValueError(f"Unknown sampling mode self.sampling={self.sampling}. Expected 1 or 2.")            

    def aggregate(self, wsolns): 
        
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

        for (w, soln) in wsolns:  # w is the number of samples
            if self.sampling == 2:
                w = 1.0

            total_weight += w 
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln


    def aggregate2(self, weights_before, Deltas, hs): 
        
        demominator = np.sum(np.asarray(hs))
        num_clients = len(Deltas)
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]

        return new_solutions

    def heuristic_sgd_update(self, weights_before, soln, loss, q, learning_rate):
        '''
        this is actually normal gradient descent if scaling factor is just learning rate
        '''
        grads = soln[1]
        grad_norm = norm_grad_sparse(grads)  # about 0.1
        q_dynamic = q
        lr = learning_rate


        scaling_factor = lr
        scaled_updates = []
        for layer in range(len(grads)):
            if layer == 0:
                indices = grads[0].indices
                values =  grads[0].values
                first_layer_dense = np.zeros((80,8))
                for i in range(indices.shape[0]):
                    first_layer_dense[indices[i], :] = values[i, :]
                scaled_updates.append(scaling_factor * first_layer_dense)
            else:
                scaled_updates.append(scaling_factor * grads[layer])
        new_solutions = [u - v for u, v in zip(weights_before, scaled_updates)]

        return (soln[0], new_solutions)


