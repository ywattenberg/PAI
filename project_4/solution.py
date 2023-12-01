import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.activation_fn = nn.ReLU()
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise NotImplementedError


        self.input = nn.Linear(input_dim, hidden_size)
        self.hidden = []
        for i in range(hidden_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
            self.hidden.append(self.activation_fn)
        self.hidden = nn.Sequential(*self.hidden)
        self.add_module('hidden', self.hidden)
           
        self.output = nn.Linear(hidden_size, output_dim)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        x = self.input(s)
        x = self.activation_fn(x)
        x = self.hidden(x)
        x = self.output(x)
        return x
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        self.network = NeuralNetwork(self.state_dim, 2*self.action_dim, self.hidden_size, self.hidden_layers, 'relu')
        self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool = False) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action, log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])


        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        out = self.network(state)
        mean, log_std = out[:,:self.action_dim], out[:,self.action_dim:]
        # print(f"mean shape: {mean.shape}, log_std shape: {log_std.shape}")
        c_log_std = self.clamp_log_std(log_std)
        
        normal = Normal(mean, c_log_std.exp())
        if deterministic:
            actions = normal.mode
        else:
            actions = normal.rsample()
        
        action = torch.tanh(actions)
        log_prob = normal.log_prob(actions)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # print(f"{action.shape}, {log_prob.shape}, input shape: {state.shape}")

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.critic_1 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.critic_1.to(self.device)
        self.critic_2 = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.critic_2.to(self.device)
        self.optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.critic_lr)
    
    def get_q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        :param state: torch.Tensor, state of the agent
        :param action: torch.Tensor, action taken by the agent
        Returns:
        :param q_value: torch.Tensor, q_value for the state action pair.
        '''
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        input = torch.cat([state, action], dim=1)
        return torch.cat([self.critic_1(input), self.critic_2(input)], dim=1)

        


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000

        ###
        self.gamma = 0.95
        self.tau = 1
        self.target_update_interval = 100
        self.lr = 3e-4
        ###

        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.
        self.target_entropy = -1*self.action_dim
        self.log_entropy_coef = torch.zeros(self.action_dim).to(self.device).requires_grad_(True)

        self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=self.lr)
        

        self.policy = Actor(256, 2, self.lr, self.state_dim, self.action_dim, self.device)
        self.critic = Critic(256, 2, self.lr, self.state_dim, self.action_dim, self.device)
        self.critic_target = Critic(256, 2, 0, self.state_dim, self.action_dim, self.device)

        self.critic_target_update(self.critic.critic_1, self.critic_target.critic_1, self.tau, False)
        self.critic_target_update(self.critic.critic_2, self.critic_target.critic_2, self.tau, False)

        # Set eval mode for critic target networks
        self.critic_target.critic_1.eval()
        self.critic_target.critic_2.eval()

        # Set up lr scheduler for all optimizers
        self.lr_scheduler = []
        self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=100, gamma=0.99))
        self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=100, gamma=0.99))
        self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.entropy_optimizer, step_size=100, gamma=0.99))

        self.updates = 0

        


    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """

        state = torch.tensor(s, dtype=torch.float32).to(self.device)
        action, _ = self.policy.get_action_and_log_prob(state, deterministic=False)
        action = action.squeeze(0).cpu().detach().numpy()
        
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        entropy_coef = torch.exp(self.log_entropy_coef)
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.get_action_and_log_prob(s_prime_batch, False)
            next_q = self.critic_target.get_q_values(s_prime_batch, next_actions)
            next_q = torch.min(next_q, dim=1, keepdim=True)[0] - entropy_coef* next_log_probs
            next_q_value = r_batch + self.gamma * next_q
        q = self.critic.get_q_values(s_batch, a_batch)
        critic_loss = 0.5 * (torch.nn.functional.mse_loss(q[:,0], next_q_value.detach()) + torch.nn.functional.mse_loss(q[:,1], next_q_value.detach()))
        self.run_gradient_update_step(self.critic, critic_loss)

        actions, log_probs = self.policy.get_action_and_log_prob(s_batch, False)
        q = self.critic.get_q_values(s_batch, actions)
        min_q, _ = torch.min(q, dim=1, keepdim=True)
        actor_loss = (entropy_coef * log_probs - min_q).mean()
        self.run_gradient_update_step(self.policy, actor_loss)

        entropy_coef_loss = -(self.log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()
        self.entropy_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.entropy_optimizer.step()



        if self.updates % self.target_update_interval == 0:
            self.critic_target_update(self.critic.critic_1, self.critic_target.critic_1, self.tau, True)
            self.critic_target_update(self.critic.critic_2, self.critic_target.critic_2, self.tau, True)

        self.updates += 1


        # actions, log_probs = self.policy.get_action_and_log_prob(s_batch)
        # log_probs = log_probs.reshape(-1, 1)
        
        # ent_coef = torch.exp(self.log_entropy_coef).to(self.device)
        # ent_coef_loss = -(self.log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()
        
        # self.entropy_optimizer.zero_grad()
        # ent_coef_loss.backward()
        # self.entropy_optimizer.step()

        # with torch.no_grad():
        #     next_actions, next_log_probs = self.policy.get_action_and_log_prob(s_prime_batch)
        #     next_q = self.critic_target.get_q_values(s_batch, next_actions)
        #     next_q = torch.min(next_q, dim=1, keepdim=True)[0] - ent_coef * next_log_probs.reshape(-1, 1)
        #     #print(f"next_q shape: {next_q}")
        #     target_q = r_batch + self.gamma * next_q

        # q_values = self.critic.get_q_values(s_batch, a_batch)

        # critic_loss = 0.5 * (torch.nn.functional.mse_loss(q_values[:,0], target_q.detach()) + torch.nn.functional.mse_loss(q_values[:,1], target_q.detach()))

        # self.critic.optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic.optimizer.step()

        # q_values = self.critic.get_q_values(s_batch, actions)
        # min_q, _ = torch.min(q_values, dim=1, keepdim=True)
        # actor_loss = (ent_coef * log_probs - min_q).mean()
        
        # self.policy.optimizer.zero_grad()
        # actor_loss.backward()
        # self.policy.optimizer.step()

        # self.critic_target_update(self.critic.critic_1, self.critic_target.critic_1, self.tau, True)

        # Update learning rate
        # for lr_scheduler in self.lr_scheduler:
        #     lr_scheduler.step()
        

        # value = self.value.network(s_batch)
        # target_value = self.target_value.network(s_prime_batch)

        # # TODO: Implement Critic(s) update here.
        # actions, log_probs = self.policy.get_action_and_log_prob(s_batch, True)
        # q1 = self.critic_1.network(torch.cat([s_batch, actions], dim=1))
        # q2 = self.critic_2.network(torch.cat([s_batch, actions], dim=1))
        # critic_value = torch.min(q1, q2)

        # self.value.optimizer.zero_grad()
        # value_target = critic_value - log_probs
        # value_loss = 0.5 * torch.nn.functional.mse_loss(value, value_target.detach())
        # value_loss.backward(retain_graph=True)
        # self.value.optimizer.step()


        # actions, log_probs = self.policy.get_action_and_log_prob(s_batch, True)
        
        # q1 = self.critic_1.network(torch.cat([s_batch, actions], dim=1))
        # q2 = self.critic_2.network(torch.cat([s_batch, actions], dim=1))
        # critic_value = torch.min(q1, q2)

        # actor_loss = (log_probs - critic_value).mean()
        # self.policy.optimizer.zero_grad()
        # actor_loss.backward(retain_graph=True)
        # self.policy.optimizer.step()

        # self.critic_1.optimizer.zero_grad()
        # self.critic_2.optimizer.zero_grad()
        # q_hat = self.reward_scale*r_batch + self.gamma * target_value
        # q1_old = self.critic_1.network(torch.cat([s_batch, a_batch], dim=1))
        # q2_old = self.critic_2.network(torch.cat([s_batch, a_batch], dim=1))
        # critic_loss_1 = 0.5 * torch.nn.functional.mse_loss(q1_old, q_hat.detach())
        # critic_loss_2 = 0.5 * torch.nn.functional.mse_loss(q2_old, q_hat.detach())

        # critic_loss = critic_loss_1 + critic_loss_2
        # critic_loss.backward()
        # self.critic_1.optimizer.step()
        # self.critic_2.optimizer.step()

        # self.critic_target_update(self.value.network, self.target_value.network, self.tau, True)


        # TODO: Implement Policy update here


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
