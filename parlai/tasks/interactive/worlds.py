#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from typing import Optional
from parlai.core.opt import Opt

from parlai.core.params import ParlaiParser
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.core.message import Message


class InteractiveWorld(DialogPartnerWorld):
    """
    Simple interactive world involving just two agents talking.

    In more sophisticated worlds the environment could supply information, e.g. in
    tasks/convai2 both agents are given personas, so a world class should be written
    especially for those cases for given tasks.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        # no default args
        return parser

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts(shared=shared)
        self.turn_cnt = 0

    def init_contexts(self, shared=None):
        """
        Override to load or instantiate contexts to be used to seed the chat.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the episode.

        This function will be called before the first turn of every episode.
        """
        return ['', '']

    def finalize_episode(self):
        print("CHAT DONE ")
        if not self.epoch_done():
            print("\n... preparing new chat... \n")

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.turn_cnt == 0:
            self.p1, self.p2 = self.get_contexts()

        acts = self.acts
        agents = self.agents
        if self.turn_cnt == 0 and self.p1 != '':
            # add the context on to the first message to agent 0
            context_act = Message(
                {'id': 'context', 'text': self.p1, 'episode_done': False}
            )
            agents[0].observe(validate(context_act))
        try:
            act = deepcopy(agents[0].act())
        except StopIteration:
            self.reset()
            self.finalize_episode()
            self.turn_cnt = 0
            return
        acts[0] = act
        if self.turn_cnt == 0 and self.p2 != '':
            # add the context on to the first message to agent 1
            context_act = Message(
                {'id': 'context', 'text': self.p2, 'episode_done': False}
            )
            agents[1].observe(validate(context_act))
        agents[1].observe(validate(act))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.turn_cnt += 1

        if act['episode_done']:
            self.finalize_episode()
            self.turn_cnt = 0

    def parley(self, response_text):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.turn_cnt == 0:
            self.p1, self.p2 = self.get_contexts()

        acts = self.acts
        agents = self.agents
        if self.turn_cnt == 0 and self.p1 != '':
            # add the context on to the first message to agent 0
            context_act = Message(
                {'id': 'context', 'text': self.p1, 'episode_done': False}
            )
            agents[0].observe(validate(context_act))
        try:
            #act = deepcopy(agents[0].act())
            act = deepcopy(agents[0].act(response_text))
        except StopIteration:
            self.reset()
            self.finalize_episode()
            self.turn_cnt = 0
            return
        acts[0] = act
        if self.turn_cnt == 0 and self.p2 != '':
            # add the context on to the first message to agent 1
            context_act = Message(
                {'id': 'context', 'text': self.p2, 'episode_done': False}
            )
            agents[1].observe(validate(context_act))
        agents[1].observe(validate(act))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.turn_cnt += 1

        if act['episode_done']:
            self.finalize_episode()
            self.turn_cnt = 0

# parlai train_model -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized -m transformer/generator --multitask-weights 1,3,3,3 --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True --model-file /home/azureuser/ankur/testmodel/model_90M/test_train_90M

# parlai train_model -t jsonfile --jsonfile-datapath "/home/azureuser/ankur/test.json" -m transformer/generator  --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True --model-file /home/azureuser/ankur/testmodel/model_90M/test_train_90M_v2 --epochs

# parlai train_model -t fromfile:parlaiformat --fromfile_datapath "/home/azureuser/ankur/aadhar.txt" -m transformer/generator  --init-model zoo:blender/blender_90M/model --dict-file zoo:blender/blender_90M/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation True -vp 15 -stim 60 -vme 20000 -bs 16 -vmt ppl -vmm min --save-after-valid True --model-file /home/azureuser/ankur/testmodel/model_90M/test_train_90M_v2 --epochs


# { "dialog": [ [  {"id": "partner1", "text": "hello how are you today?"},  
#                 {"id": "partner2", "text": "i'm great thanks! what are you doing?"},  
#                 {"id": "partner1", "text": "i've just been bikinig."},        
#                 {"id": "partner2", "text": "oh nice, i haven't got on a bike in years!"} ] ]}