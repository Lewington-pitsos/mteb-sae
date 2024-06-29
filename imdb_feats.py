from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformer_lens.utils import tokenize_and_concatenate
from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData
from datasets import load_dataset

device = 'cuda'
model = HookedTransformer.from_pretrained("gpt2-small", device = device)

# the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
# Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
# We also return the feature sparsities which are stored in HF for convenience. 
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
    device = device
)


dataset = load_dataset("imdb")
subsets = {}
for label in set(dataset['train']['label']):  # Assuming you are working with the 'train' split
    subsets[label] = dataset['train'].filter(lambda example: example['label'] == label)

for k, subset in subsets.items():
    dataset = subset

    token_dataset = tokenize_and_concatenate(
        dataset= dataset,# type: ignore
        tokenizer = model.tokenizer, # type: ignore
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    feature_vis_config_gpt = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=None,
        batch_size=2048,
        minibatch_size_tokens=128,
        verbose=True,
    )

    sae_vis_data_gpt = SaeVisData.create(
        encoder=sae,
        model=model, # type: ignore
        tokens=token_dataset["tokens"],  # type: ignore
        cfg=feature_vis_config_gpt,
    )

    sae_vis_data_gpt.save_json(f"all-{str(k)}.json")




#     Forward passes to cache data for vis:   3%|███▋                                                                                                                 | 48/1536 [21:33<36:56,  1.49s/it]
# Extracting vis data from cached data:   2%|██▌                                                                                                             | 556/24576 [23:25<17:Extracting vis daForward passes to cache data for vis: