FRAMEWORK = "pytorch"                   
MODEL_PATH = "model/model_T10.pt"       
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224

TEAM_ID = "T10"
MAX_UPLOAD_MB = 10                      # reject uploads larger than this
ALLOWED_FORMATS = {"JPEG", "PNG"}       # checked from the file bytes, not the name
