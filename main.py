from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import shap
from schemas import PatientData

app = FastAPI(title="Clinical Readmission API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# globals
model = None
threshold = 0.5
explainer = None
preprocessor = None

@app.on_event("startup")
async def load_assets():
    global model, threshold, explainer, preprocessor
    try:
        # load serialized artifacts
        model = joblib.load("clinical_readmission_model.joblib")
        
        with open("optimal_threshold.txt", "r") as f:
            threshold = float(f.read().strip())
            
        # extract steps for shap
        preprocessor = model.named_steps['preprocessor']
        clf = model.named_steps['classifier']
        
        # init tree explainer
        explainer = shap.TreeExplainer(clf)
        print("ml assets loaded successfully")
    except Exception as e:
        print(f"error loading assets: {e}")
        raise e

@app.post("/predict_risk")
async def predict(data: PatientData):
    if not model:
        raise HTTPException(status_code=503, detail="model not initialized")
        
    try:
        # convert payload to dataframe
        df = pd.DataFrame([data.model_dump()])
        
        # get probability
        probs = model.predict_proba(df)
        pos_prob = float(probs[0][1])
        
        # risk bucketing
        if pos_prob >= 0.80:
            risk = "Critical Risk"
            intervention = True
        elif pos_prob >= threshold:
            risk = "Elevated Risk"
            intervention = True
        else:
            risk = "Standard Care"
            intervention = False
            
        # calculate shap values for local explainability
        processed_features = preprocessor.transform(df)
        shap_vals = explainer.shap_values(processed_features)
        feature_names = preprocessor.get_feature_names_out()
        
        # zip features with their shap values
        shap_dict = dict(zip(feature_names, [float(x) for x in shap_vals[0]]))
        
        # get top 5 driving factors
        top_5 = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

        return {
            "probability": round(pos_prob, 4),
            "risk_tier": risk,
            "needs_intervention": intervention,
            "threshold_used": round(threshold, 4),
            "top_shap_factors": top_5
        }
        
    except Exception as e:
        print(f"prediction error: {e}")
        raise HTTPException(status_code=500, detail="internal server error")
