import mteb

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = mteb.get_model(model_name)

tasks = mteb.get_tasks(tasks=['CQADupstackAndroidRetrieval'])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")