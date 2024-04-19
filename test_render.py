from pettingzoo.mpe import simple_reference_v3, simple_adversary_v3, simple_push_v3, simple_v3, simple_spread_v3, simple_speaker_listener_v4
from MADDPG import MADDPG
import numpy as np
import pandas as pd
from argParser import parse_args
test_env = False
env_class= simple_speaker_listener_v4#simple_adversary_v3#simple_spread_v3 if test_env else simple_speaker_listener_v4
env_name = "simple_speaker_listener_v4"#"simple_adversary_v3" #simple_spread_v3" if test_env else "simple_speaker_listener_v4"

env = env_class.parallel_env(max_cycles = 25,  render_mode="human",continuous_actions=True)#render_mode="human",


INFERENCE = True
PRINT_INTERVAL = 1000
MAX_EPISODES = 300000
BATCH_SIZE = 1024
MAX_STEPS = 25
total_steps = 0
score = -10
best_score = 0
score_history = []
# env = simple_adversary_v3.env()
# obs = env.reset(seed=110)
observations, infos = env.reset(seed=111)
# print(env)
# print("num agents ", env.num_agents)
# print("observation space ", env.observation_spaces)
# print("action space ", env.action_spaces)
# print(obs)
args = parse_args()
if env_name == "simple_speaker_listener_v4":
    params = "_20k_lr0_0001_a_64_c_128_sig_il"#"_lr0_001_5k_ep_a_32_c_128_newR"
    args.out_act ="sigmoid"
    args.critic_hidden = 128
    args.actor_hidden = 64
if env_name == "simple_spread_v3":
    params = "_lr0_0001_150k_ep_bs256_a_64_c_256_newR_newN"#"_lr0_001_10k_ep_bs128_a_128_c_128_newR"#"_lr0_001_bs512_200learn"#100k
    args.critic_hidden = 256
    args.actor_hidden = 64#64
if env_name == "simple_reference_v3":
    params = "_lr0_001_4k_ep"
if env_name == "simple_adversary_v3":
    params = "_seed10_50k_k4_bs64_newN"#"_seed10_k1_10k_bs256"#"_seed2_k4_10k_bs64"
    # args.critic_hidden = 128
    # args.actor_hidden = 128
    
n_agents = env.num_agents
actor_dims = []
# for i in range(n_agents):
#     actor_dims.append(env.observation_spaces[list(env.observation_spaces.keys())[i]].shape[0])  

# critic_dims = sum(actor_dims)

actor_dims = []
action_dim = []
for i in range(n_agents):
    actor_dims.append(env.observation_space(env.agents[i]).shape[0])#s[list(env.observation_spaces.keys())[i]].shape[0])  
    action_dim.append(env.action_space(env.agents[i]).shape[0])#s[list(env.action_spaces.keys())[i]].shape[0])
critic_dims = sum(actor_dims)
# action_dim = env.action_space(env.agents[0]).shape[0]
maddpg = MADDPG(actor_dims, critic_dims+sum(action_dim), n_agents, action_dim,chkpt_dir="nets", scenario=f"/{env_name}{params}", args=args)

if INFERENCE:
    maddpg.load_checkpoint()

tot_actions = []
for i in range(5):
    k = np.random.randint(0, args.sub_policy)
    observations, infos = env.reset(seed=51+i)
    while env.agents:
        observations = list(observations.values())
        # this is where you would insert your policy
        actions = maddpg.choose_action(observations, k=k, eval=INFERENCE,ep=i, max_ep=MAX_EPISODES)
        
        actions = {agent:action.reshape(-1) for agent, action in zip(env.agents, actions)}
        # print(observations)
        # print(actions)
        tot_actions.append(actions)
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
 
# df = pd.DataFrame(tot_actions)
# df.to_csv(f"{env_name}_actions.csv")




# sl_env_out = args.out_act_ls if agent_idx else args.out_act_sp
#         out_act_string = sl_env_out if args.env_id == "simple_speaker_listener_v4" else args.out_act
#         self.actor = [Actor(actor_dims, n_actions, self.name+'_actor_policy'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate, hidden_dim=args.actor_hidden, out_act_string = out_act_string) for i in range(args.sub_policy)]
#         self.critic = [Critic(critic_dims, n_actions, n_agents=n_agents, name=self.name+'_critic'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate,hidden_dim=args.critic_hidden) for i in range(args.sub_policy)]

#         self.target_actor = [Actor(actor_dims, n_actions, self.name+'_target_actor'+str(i), chkpt_dir=chkpt_dir,lr=args.learning_rate,hidden_dim=args.actor_hidden, out_act_string = out_act_string)for i in range(args.sub_policy)]
