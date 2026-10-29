
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Mock Customer Application API",
    description="A demo API for querying sample loan applications.",
    version="1.0.0",
)

APPLICATIONS = {
    "APP-1001": {
        "application_id": "APP-1001",
        "status": "approved",
        "description": "The application has been approved.",
    },
    "APP-1002": {
        "application_id": "APP-1002",
        "status": "under_review",
        "description": "The application is currently under review.",
    },
    "APP-1003": {
        "application_id": "APP-1003",
        "status": "rejected",
        "description": "The application has been rejected.",
    },
}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/applications/{application_id}")
def get_application(application_id: str):
    application = APPLICATIONS.get(application_id.upper())

    if application is None:
        raise HTTPException(
            status_code=404,
            detail=f"Application '{application_id}' was not found.",
        )

    return application
