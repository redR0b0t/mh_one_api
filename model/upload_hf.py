from huggingface_hub import HfApi
api = HfApi()

# model_path="/workspaces/mh_one_api/model/ft_models/flan-t5-xl_peft_ft_v2/checkpoint-239100"
# model_path="/workspaces/mh_one_api/model/ft_models/ft_final"
# api.upload_folder(
#     folder_path=model_path,
#     repo_id="blur0b0t/mh_one_api",
#     repo_type="model",
# )


webapp_path="/workspaces/mhi_pred_app/build/web"
api.upload_folder(
    folder_path=webapp_path,
    repo_id="blur0b0t/mh_one_api",
    repo_type="space",
)