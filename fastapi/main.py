from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

posts: list[dict] = [
    {
        "id": 1,
        "author": "Corey Schafer",
        "title": "FastAPI is Awesome",
        "content": "This framework is really easy to use and super fast.",
        "date_posted": "April 20, 2025",
    },
    {
        "id": 2,
        "author": "Jane Doe",
        "title": "Python is Great for Web Development",
        "content": "Python is a great language for web development, and FastAPI makes it even better.",
        "date_posted": "April 21, 2025",
    },
]

@app.get("/", response_class=HTMLResponse, include_in_schema=False) # include_in_schema=False hides this endpoint from the OpenAPI docs
# response_class=HTMLResponse tells FastAPI to return an HTML response instead of JSON
@app.get("/posts", response_class=HTMLResponse, include_in_schema=False)
def home():
    return f"<h1>Welcome to the FastAPI Blog!</h1><p>We have {len(posts)} posts.</p>"

@app.get("/posts", response_class=JSONResponse)
def get_posts():
    return {"posts": posts}

