import os 
from random import shuffle

import numpy as np
import torch
import time

from carle.env import CARLE
from carle.mcl import CornerBonus, SpeedDetector, PufferDetector, AE2D, RND2D
from game_of_carle.agents.harli import HARLI

import bokeh
import bokeh.io as bio
from bokeh.io import output_notebook, show, curdoc
from bokeh.plotting import figure

from bokeh.layouts import column, row
from bokeh.models import TextInput, Button, Paragraph
from bokeh.models import ColumnDataSource

from bokeh.events import DoubleTap, Tap

import matplotlib.pyplot as plt
my_cmap = plt.get_cmap("viridis")

device_string = "cpu"

obs_dim = 32
act_dim = 6

agent =  HARLI(device=device_string, obs_dim=obs_dim, act_dim=act_dim)

policy_list = []

directory_list = os.listdir("./policies/")

for filename in directory_list:
    
    if "HARLI" in filename and "glider" in filename:
        
        policy_list.append(os.path.join(".", "policies", filename))
        
policy_list.sort()

# instantiate CARLE with a speed detection wrapper
env = CARLE(height=obs_dim, width=obs_dim, action_height=act_dim, action_width=act_dim, device=device_string)
env = SpeedDetector(env)
        
agent.set_params(np.load(policy_list[0]))


def agent_off():

    global agent_on
    global rule_index
    global my_period
    global p

    my_period = 512
    agent_on = False

    rule_index = 0
    reset_next_ruleset()

    button_go.label == "Run >"

    curdoc().add_root(control_layout)
    message.text = "Human's turn!"


def summary_screen():
    global p_bar

    curdoc().remove_root(display_layout)
    curdoc().remove_root(button_go)
    curdoc().remove_root(control_layout)
    curdoc().remove_root(rule_layout)
    curdoc().remove_root(message_layout)

    if button_go.label != "Run >":

        curdoc().remove_periodic_callback(curdoc().session_callbacks[0])
        button_go.label = "Run >"

    my_width = 0.25
    x = [0, 2, 4 ]
    x2 = [elem+my_width*2 for elem in x]

    harli_top = [elem[1] / max([100, elem[2]]) for elem in harli_scores]
    human_top = [elem[1] / max([100, elem[2]]) for elem in human_scores]


    p_bar = figure(plot_width=3*256, plot_height=3*256, title="BAR PLOT!")

    p_bar.vbar(x, width=0.5, top=harli_top, color=[[elem * 255 for elem in my_cmap(.35)]]*3, legend_label="BOT") 
    p_bar.vbar(x2, width=0.5, top=human_top,color=[[elem * 255 for elem in my_cmap(.75)]]*3, legend_label="HMN                         ")
    p_bar.vbar([6], width=0, top=0.0)

    curdoc().add_root(p_bar)

    curdoc().add_root(message1)
    curdoc().add_root(message2)
    curdoc().add_root(message3)
    curdoc().add_root(message4)
    curdoc().add_root(message5)
    curdoc().add_root(message6)

    human_average = np.mean([elem[1] / max([100, elem[2]]) for elem in human_scores])
    harli_average = np.mean([elem[1] / max([100, elem[2]]) for elem in harli_scores])

    message1.text = "**************************************************** REWARDS per step ****************************************************"
    message2.text = "__rules___________________________________________human_____________________________________HARLI__"
    message3.text = f"{harli_scores[0][0]}_____________________________________{human_scores[0][1] / max([100, human_scores[0][2]]):.4f}"\
            f"_____________________________________{harli_scores[0][1] / max([100, harli_scores[0][2]]):.4f}"
    message4.text = f"{harli_scores[1][0]}_____________________________________{human_scores[1][1] / max([100, human_scores[1][2]]):.4f}"\
            f"_____________________________________{harli_scores[1][1] / max([100, harli_scores[1][2]]):.4f}"
    message5.text = f"{harli_scores[2][0]}_____________________________________{human_scores[2][1] / max([100, human_scores[2][2]]):.4f}"\
            f"_____________________________________{harli_scores[2][1] / max([100, harli_scores[2][2]]):.4f}"
    message6.text = f"Average__________________________________________{human_average:.4f}_____________________________________{harli_average:.4f}"


    curdoc().add_root(button_start_over)

def update():
    global obs
    global stretch_pixel
    global action
    global agent_on
    global my_step
    global max_steps
    global rule_index
    global rewards
    global reward_sum


    if agent_on and rule_index == len(rules) and my_step >= max_steps:

        harli_scores.append((rules[rule_index-1], reward_sum, my_step))
        agent_off()

    elif not(agent_on) and rule_index == len(rules) and my_step >= max_steps:

        human_scores.append((rules[rule_index-1], reward_sum, my_step))
        summary_screen()

    elif agent_on and my_step >= max_steps:

        harli_scores.append((rules[rule_index-1], reward_sum, my_step))
        reset_next_ruleset()

    elif not(agent_on) and my_step >= max_steps:

        human_scores.append((rules[rule_index-1], reward_sum, my_step))
        reset_next_ruleset()

    else:

        obs, r, d, i = env.step(action)
        rewards = np.append(rewards, r.cpu().numpy().item())
        if agent_on:
            action = agent(obs) 
        else:
            action = torch.zeros_like(action)

        padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()

        my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
        my_img[my_img > 3.0] = 3.0
        (padded_action*2 + obs.squeeze()).cpu().numpy()

        new_data = dict(my_image=[my_img])

        my_weights = agent.get_weights().reshape(dim_wh, dim_ww)
        new_weights = dict(my_image=[my_weights])

        #new_line = dict(x=np.arange(my_step+2), y=rewards)
        new_line = dict(x=[my_step], y=[r.cpu().numpy().item()])

        source.stream(new_data, rollover=1)
        source_plot.stream(new_line, rollover=2000)

        source_weights.stream(new_weights, rollover=1)

        my_step += 1
        reward_sum += r.item()

        turn_msg = "Bot's turn: " if agent_on  else "Human's turn: "
        message.text = f"{turn_msg}step {my_step}, reward: {r.item():.4f}, mean reward per step: {(reward_sum/my_step):.4f} \n"\
                f" rule index = {rule_index}"

def go():

    if button_go.label == "Run >":
        my_callback = curdoc().add_periodic_callback(update, my_period)
        button_go.label = "Pause"
        #curdoc().remove_periodic_callback(my_callback)

    else:
        curdoc().remove_periodic_callback(curdoc().session_callbacks[0])
        button_go.label = "Run >"

    curdoc().remove_root(button_start)

    curdoc().remove_root(message1)
    curdoc().remove_root(message2)
    curdoc().remove_root(message3)
    curdoc().remove_root(message4)
    curdoc().remove_root(message5)


    curdoc().add_root(control_layout)

def reset_next_ruleset():

    global obs
    global action
    global stretch_pixel
    global my_step
    global rewards
    global use_spaceship
    global rewards
    global reward_sum
    global rule_index
    global human_scores
    global harli_scores


    reward_sum = 0.0

    my_step = 0
    new_line = dict(x=[my_step], y=[0])
    obs = env.reset()
    stretch_pixel = torch.zeros_like(obs).squeeze()
    stretch_pixel[0,0] = 3
    agent.reset()

    if agent_on:
        action = agent(obs) 
    else:
        action = torch.zeros_like(action)

    padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()

    my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
    my_img[my_img > 3.0] = 3.0
    (padded_action*2 + obs.squeeze()).cpu().numpy()
    new_data = dict(my_image=[my_img])


    my_weights = agent.get_weights().reshape(dim_wh, dim_ww)
    new_weights = dict(my_image=[my_weights])


    source.stream(new_data, rollover=1)
    source_plot.stream(new_line, rollover=2)

    source_weights.stream(new_weights, rollover=1)

    message.text = f"step {my_step} \n"\
            f"{policy_list[0]}"

    rewards = np.array([0])

    source_plot.stream(new_line, rollover=1)
    source.stream(new_data, rollover=8)

    env.rules_from_string(rules[rule_index])
    rule_index += 1

    if (button_go.label == "Pause") and not(agent_on) and rule_index != 0:
        go()



def start():
    global my_period
    global my_step
    global rule_index
    global max_steps
    global human_scores
    global harli_scores
    global rules
    global agent_on

    rule_index = 0
    max_steps = 256

    agent_on = True

    harli_scores = []
    human_scores = []

    rules = ["B3___/S23___", \
        "B368_/S245__", \
        "B3678/S34678"]

    shuffle(rules)

    curdoc().remove_root(message1)
    curdoc().remove_root(message2)
    curdoc().remove_root(message3)
    curdoc().remove_root(message4)
    curdoc().remove_root(message5)
    curdoc().remove_root(message6)
    curdoc().remove_root(p_bar)

    curdoc().add_root(display_layout)
    curdoc().remove_root(button_start_over)

    #curdoc().add_root(rule_layout)
    curdoc().add_root(message_layout)

    curdoc().add_root(message1)
    curdoc().add_root(message2)
    curdoc().add_root(message3)
    curdoc().add_root(message4)
    curdoc().add_root(message5)

    reset_next_ruleset()

    message.text = ""
    message1.text = f"Can you beat the bot?"
    message2.text = f" You are tasked with maximizing the mean reward displayed in the top right by clicking cells in the {act_dim} by {act_dim} 'action space' in the center of the grid on the left."
    message3.text = f" The grid universe will cycle through {len(rules)} different shuffled rulesets, your only guide is the reward signal!"
    message4.text = f" Your adversary has never trained on these rulesets either, and starts from a random initialization of weights."
    message5.text = f" Click 'Go!' to get started, HARLI will go first. ___ (I'm sorry this text is small)"

    curdoc().add_root(button_start)


def faster():


    global my_period
    my_period = max([my_period * 0.5, 32])
    go()
    go()

def slower():

    global my_period
    my_period = min([my_period * 2, 8192])
    go()
    go()

def human_toggle(event):
    global action

    if not(agent_on):
        coords = [np.round(env.height*event.y/256-0.5), np.round(env.width*event.x/256-0.5)]
        offset_x = (env.height - env.action_height) / 2
        offset_y = (env.width - env.action_width) / 2

        coords[0] = coords[0] - offset_x
        coords[1] = coords[1] - offset_y

        coords[0] = np.uint8(np.clip(coords[0], 0, env.action_height-1))
        coords[1] = np.uint8(np.clip(coords[1], 0, env.action_height-1))

        action[:, :, coords[0], coords[1]] = 1.0 * (not(action[:, :, coords[0], coords[1]]))

        #padded_action = stretch_pixel/2 + env.action_padding(action).squeeze()
        padded_action = stretch_pixel/2 + env.inner_env.action_padding(action).squeeze()

        my_img = (padded_action*2 + obs.squeeze()).cpu().numpy()
        my_img[my_img > 3.0] = 3.0
        (padded_action*2 + obs.squeeze()).cpu().numpy()
        new_data = dict(my_image=[my_img])

        source.stream(new_data, rollover=8)


global obs
global my_period
global agent_on
global action
global reward_sum
global max_steps
global rule_index
global human_scores
global harli_scores

dim_wh = 24
dim_ww = 24

obs = env.reset()
my_weights = agent.get_weights().reshape(dim_wh, dim_ww)

p = figure(plot_width=3*256, plot_height=3*256, title="CA Universe")
p_plot = figure(plot_width=int(1.25*256), plot_height=int(1.25*256), title="'Reward'")
p_weights = figure(plot_width=int(1.255*256), plot_height=int(1.25*256), title="Weights")
p_bar = figure(plot_width=3*256, plot_height=3*256, title="BAR PLOT!")

reward_sum = 0.0

my_period = 32

agent_on = True
action = torch.zeros(1, 1, env.action_height, env.action_width)

source = ColumnDataSource(data=dict(my_image=[obs.squeeze().cpu().numpy()]))
source_plot = ColumnDataSource(data=dict(x=np.arange(1), y=np.arange(1)*0))

source_weights = ColumnDataSource(data=dict(my_image=[my_weights]))

img = p.image(image='my_image',x=0, y=0, dw=256, dh=256, palette="Magma256", source=source)
line_plot = p_plot.line(line_width=3, color="firebrick", source=source_plot)

img_w = p_weights.image(image='my_image',x=0, y=0, dw=240, dh=240, palette="Magma256", source=source_weights)

button_go = Button(sizing_mode="stretch_width", label="Run >")     
button_slower = Button(sizing_mode="stretch_width",label="<< Slower")
button_faster = Button(sizing_mode="stretch_width",label="Faster >>")

button_next_ruleset = Button(sizing_mode="stretch_width", label="Next ruleset")
button_start = Button(sizing_mode="stretch_width", label="Go!")

button_start_over = Button(sizing_mode="stretch_width", label="Go again!")


message = Paragraph(default_size=900)

message1 = Paragraph(default_size=900)
message2 = Paragraph(default_size=900)
message3 = Paragraph(default_size=900)
message4 = Paragraph(default_size=900)
message5 = Paragraph(default_size=900)

message6 = Paragraph(default_size=900)

p.on_event(Tap, human_toggle)

button_start_over.on_click(start)
button_go.on_click(go)
button_start.on_click(go)
button_faster.on_click(faster)
button_slower.on_click(slower)
button_next_ruleset.on_click(reset_next_ruleset)

control_layout = row(button_slower, button_go, button_faster)

rule_layout = row(button_next_ruleset)

display_layout = row(p, column(p_plot, p_weights))
message_layout = row(message)


start()


