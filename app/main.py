from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Fake News Detector API Running"}