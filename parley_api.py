from flask import Flask, request
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent

opt = {'init_opt': None, 'allow_missing_init_opts': False, 
       'task': 'blended_skill_talk', 'download_path': None, 'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 'verbose': False, 'is_debug': False, 'datapath': '/home/azureuser/ankur/ParlAI/data', 'model': None, 
       'model_file': '/home/azureuser/ankur/ParlAI/data/models/blender/blender_3B/model', 'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 'display_add_fields': '', 
       'interactive_task': True, 
       'outfile': '', 'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'mutators': None, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'beam_size': 1, 'beam_min_length': 1, 'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': True, 'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 'compute_tokenized_bleu': False, 
       'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'safe', 'force_fp16_tokens': False, 'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'bpe_dropout': None, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 
       'display_partner_persona': False, 
       'include_personas': False, 'include_initial_utterances': False, 'safe_personas_only': False, 'parlai_home': '/home/azureuser/ankur/ParlAI', 'override': {}, 'starttime': 'Jun16_17-44'}


#Create Model Agent, Human Agent and the World
agent = create_agent(opt, requireModelExists=True)
human_agent = LocalHumanAgent(opt)
world = create_task(opt, [human_agent, agent])


app = Flask(__name__)
# run_with_ngrok(app)

@app.route("/", methods = ['GET'])
def home():
	html_content = '''<h1>Hello World!!</h1>
                      <h2>Nothing else here. Move to /parley_text</h2>'''
	return html_content


@app.route('/parley_text/', methods=['GET', 'POST'])
def welcome():
    # if request.method == 'GET':
    #     return "Get Request"
    if request.method == 'POST':
        response_text = request.values.get("Response")
        restart = request.values.get("Restart")
        if not world.epoch_done():
            if restart == "False":
                world.parley(response_text)
                teacher, response = world.get_acts()
                return response.get('text', 'No response')
            elif restart == "True":
                world.parley("[DONE]")
                return "Chat Reset"
        else:
            return "Not a Post Request"


if __name__ == '__main__':
    # model_agent, human_agent = create_model_and_human_agents(opt)
    # world = create_world(model_agent, human_agent)
    app.run(host='0.0.0.0', debug = False, port=5000)
    # app.run()