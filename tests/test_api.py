from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "status" in response.json()
