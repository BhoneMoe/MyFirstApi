from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pydantic import BaseModel
import pandas as pd
import uvicorn

# Load the trained model
model = joblib.load('my_linear_model_house_froHouse.pkl')

# Define the input data model

class HouseFeatures(BaseModel):
    OverallQual: int      # အိမ်ရဲ့ အရည်အသွေး (၁ ကနေ ၁၀ အထိ ပေးလေ့ရှိလို့ int)
    GrLivArea: int        # အထက်ထပ် နေထိုင်နိုင်သော ဧရိယာ (Square feet)
    GarageCars: int       # ကားဂိုဒေါင်ထဲ ဆံ့တဲ့ ကားအရေအတွက်
    TotalBsmtSF: float    # မြေအောက်ထပ် စုစုပေါင်း ဧရိယာ (float က ပိုသေချာပါတယ်)
    YearBuilt: int        # တည်ဆောက်ခဲ့သော ခုနှစ်
    FullBath: int         # ရေချိုးခန်း အရေအတွက်
    BedroomAbvGr: int     # အိပ်ခန်း အရေအတွက်
    LotArea: int          # မြေကွက် အကျယ်အဝန်း

# App object ကို တည်ဆောက်တာပါ
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "မင်္ဂလာပါ! ဒါက ကျွန်တော့်ရဲ့ ပထမဆုံး FastAPI ပါ"}

@app.post("/predict")
def predict(data: HouseFeatures):
     # Convert input data to a list
    input_data = pd.DataFrame([data.model_dump()])
     # Make prediction
    prediction = model.predict(input_data)
    output = round(prediction[0], 2)
    return {"prediction": prediction[0]}
