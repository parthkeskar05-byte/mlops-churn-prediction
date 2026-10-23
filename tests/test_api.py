from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_route_is_registered():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/predict" in r.json()["paths"]
