import mteb
import torch

from models import SAEEncoder

# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# model = mteb.get_model(model_name)

model_name = 'gpt2'
sae_release = 'gpt2-small-res-jb'
sae_id = 'blocks.8.hook_resid_pre'
max_seq_len = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SAEEncoder(transformer_name=model_name, max_seq_len=max_seq_len, sae_release=sae_release, sae_id=sae_id, device=device)

tasks = mteb.get_tasks(tasks=['CQADupstackAndroidRetrieval'])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}-{max_seq_len}", batch_size=256, overwrite_results=True)