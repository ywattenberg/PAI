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
        self.add_module('hidden_layers', self.hidden)
           
        self.output = nn.Linear(hidden_size, output_dim)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        x = self.input(s)
        x = self.activation_fn(x)
        x = self.hidden(x)
        x = self.output(x)
        return x
    
# self,hidden_size: int, hidden_layers: int, actor_lr: float, state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')
class GSDE:
    def __init__( self, hidden_size: int, hidden_layers: int, actor_lr: float, state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu'), log_std_init=-3.0,  batch_size=200):
        self.hidden_dim = hidden_size
        self.log_std_init = log_std_init
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.setup_gsde()

    def setup_gsde(self):
        self.feature_net = NeuralNetwork(self.state_dim, self.hidden_dim, self.hidden_layers, 2, 'relu').to(self.device)
        self.mean = nn.Linear(self.hidden_dim, self.action_dim).to(self.device)
        self.log_std_net = torch.ones(self.hidden_dim).to(self.device)
        self.log_std_net = nn.Parameter(self.log_std_net * self.log_std_init, requires_grad=True)
        self.reset_noise()
        self.optimizer = optim.Adam(list(self.feature_net.parameters()) + list(self.mean.parameters()) + [self.log_std_net], lr=self.actor_lr)
    
    def get_std(self):
        return self.log_std_net.exp()

    def reset_noise(self):
        std = self.get_std()
        self.w_dist = Normal(torch.zeros_like(std), std)
        self.exploration_noise = self.w_dist.rsample()
        self.batch_exploration_noise = self.w_dist.rsample((self.batch_size,))

    def get_action_and_params(self, state: torch.Tensor):
        latent = self.feature_net(state)
        return self.mean(latent), self.log_std_net, latent # mu, log_std, sde
    
    
    def get_noise(self, latent_sde):
        if len(latent_sde) == 1 or len(latent_sde) != len(self.batch_exploration_noise):
            return torch.dot(latent_sde, self.exploration_noise)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = (latent_sde * self.batch_exploration_noise).sum(dim=-1)
        return noise

    def get_action_and_log_prob(self, state, deterministic):
        actions, log_std, latent = self.get_action_and_params(state)

        variance = (latent**2).dot(self.get_std()**2) + 1e-6
        distribution = Normal(actions, variance.sqrt())

        if deterministic:
            actions = distribution.mean
            action = torch.tanh(actions)
        else:
            noise = self.get_noise(latent)
            actions = distribution.mean + noise
            action = torch.tanh(actions)
        g_actions = self.inv(action)
        
        log_prop = self.w_dist.log_prob(g_actions) 
        if len(log_prop.shape) > 1:
            log_prop = log_prop.sum(dim=1)
        else:
            log_prop = log_prop.sum()

        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        log_prop -= torch.sum(torch.log(1-torch.tanh(g_actions) + 1e-6), dim=-1)
        return action, log_prop

    @staticmethod
    def inv(x):
        x = x.clamp(min=-1 + 1e-6, max=1 - 1e-6)
        return 0.5 * (x.log1p() - (-x).log1p())
    
    @staticmethod
    def log_prop_corr(x):
        return torch.log(1 - torch.tanh(x)**2 + 1e-6)
    
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
        self.network = NeuralNetwork(self.state_dim, self.hidden_size, self.hidden_size, self.hidden_layers-1, 'relu').to(self.device)
        self.mean = nn.Linear(self.hidden_size, self.action_dim).to(self.device)
        self.log_std = nn.Linear(self.hidden_size, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.xavier_uniform_(self.log_std.weight)

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
        action, log_prob = torch.zeros(state.shape[0]).to(self.device), torch.ones(state.shape[0]).to(self.device)


        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        out = self.network(state)
        mean, log_std = self.mean(out), self.log_std(out)
        # print(f"mean shape: {mean.shape}, log_std shape: {log_std.shape}")
        c_log_std = self.clamp_log_std(log_std)
        
        normal = Normal(mean, c_log_std.exp())
        if deterministic:
            actions = normal.mode
        else:
            actions = normal.rsample()
        
        action = torch.tanh(actions)
        log_prob = normal.log_prob(actions) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # print(f"{action.shape}, {log_prob.shape}, input shape: {state.shape}")

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1, device: torch.device = torch.device('cpu')):
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
        return self.critic_1(input), self.critic_2(input)

        


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

# 'gamma': 0.9961269343474698, 'tau': 0.11289778101781192, 'lr': 0.007530678254384954, 'lr_step_size': 775, 'lr_gamma': 0.1720254528433003}
class Agent:
    def __init__(self, gamma: float = 0.996, tau: float = 0.113, lr: float = 0.00753, lr_step_size: int = 775, lr_gamma: float = 0.172):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000

        ###
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
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
        self.target_entropy = torch.tensor(-1*self.action_dim).to(self.device)
        self.log_entropy_coef = torch.zeros(self.action_dim).to(self.device).requires_grad_(True)
        
        self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=self.lr)

        #self.log_entropy_coef = torch.tensor(self.target_entropy)

        self.policy = GSDE(256, 2, self.lr, self.state_dim, self.action_dim, self.device)
        self.critic = Critic(256, 2, self.lr, self.state_dim, self.action_dim, self.device)
        self.critic_target = Critic(256, 2, 0, self.state_dim, self.action_dim, self.device)

        self.critic_target_update(self.critic.critic_1, self.critic_target.critic_1, self.tau, False)
        self.critic_target_update(self.critic.critic_2, self.critic_target.critic_2, self.tau, False)

        # Set up lr scheduler for all optimizers
        self.lr_scheduler = []
        self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma))
        self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma))
        self.lr_scheduler.append(torch.optim.lr_scheduler.StepLR(self.entropy_optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma))

        self.updates = 0


    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """

        state = torch.tensor(s, dtype=torch.float32).to(self.device)
        action, _ = self.policy.get_action_and_log_prob(state, deterministic=not train)
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

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.get_action_and_log_prob(s_prime_batch, False)
            next_q_t_1, next_q_t_2 = self.critic_target.get_q_values(s_prime_batch, next_actions)
            next_q = torch.min(next_q_t_1, next_q_t_2) - self.log_entropy_coef.exp()* next_log_probs
            next_q_value = r_batch + self.gamma * next_q
        q1, q2 = self.critic.get_q_values(s_batch, a_batch)
        critic_loss = (torch.nn.functional.mse_loss(q1, next_q_value) + torch.nn.functional.mse_loss(q2, next_q_value))
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        action, log_probs = self.policy.get_action_and_log_prob(s_batch, False)
        q1, q2 = self.critic.get_q_values(s_batch, action)
        min_q = torch.min(q1, q2)
        actor_loss = (self.log_entropy_coef.exp() * log_probs - min_q).mean()

        self.policy.optimizer.zero_grad()
        actor_loss.backward()
        self.policy.optimizer.step()

        entropy_coef_loss = -(self.log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()

        self.entropy_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.entropy_optimizer.step()

        self.critic_target_update(self.critic.critic_1, self.critic_target.critic_1, self.tau, True)
        self.critic_target_update(self.critic.critic_2, self.critic_target.critic_2, self.tau, True)

        self.updates += 1

        # Update learning rate
        # for lr_scheduler in self.lr_scheduler:
        #     lr_scheduler.step()


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
