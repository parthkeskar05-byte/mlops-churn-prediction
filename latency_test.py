import time, requests
URL = "http://127.0.0.1:8000/predict"
sample = {"data": {"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":12,"MonthlyCharges":65.0,"TotalCharges":780.0,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"Yes","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}}
lat = []
for _ in range(100):
    t = time.time(); r = requests.post(URL, json=sample); assert r.status_code == 200
    lat.append((time.time()-t)*1000)
lat.sort(); print("p50 ms:", round(lat[49],1)); print("p95 ms:", round(lat[94],1))
