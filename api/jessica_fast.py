from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
from typing import Optional, List

from shoppingassistant.vision import similar_items, build_chroma_collection

app = FastAPI(
    title="Shopping Assistant API",
    description="Image-based product recommendation service",
    version="1.0.0",
)


# ----------------------------
# Health & Admin
# ----------------------------

@app.get("/")
def root():
    return {"status": "Shopping Assistant running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/build-index")
def build_index(batch_size: int = 64):
    """
    Build or update the ChromaDB image index.
    Call once during setup or when new products are added.
    """
    build_chroma_collection(batch_size=batch_size)
    return {"status": "index built"}


# ----------------------------
# Image Similarity Search
# ----------------------------

@app.post("/recommend/image")
def recommend_from_image(
    image: UploadFile = File(...),
    n: int = Query(5, ge=1, le=50),
    subcategory: Optional[str] = Query(None),
    gender: Optional[str] = Query(None),
):
    """
    Upload an image and receive visually similar products.
    Optional filters: subcategory, gender.
    """

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        results = similar_items(
            image_path=tmp_path,
            n=n,
            subcategory=subcategory,
            gender=gender,
        )

        return {
            "query_image": image.filename,
            "results": results,
        }

    finally:
        os.remove(tmp_path)


# ----------------------------
# Debug / Local Testing
# ----------------------------

@app.get("/recommend/local")
def recommend_from_local_path(
    image_path: str,
    n: int = 5,
    subcategory: Optional[str] = None,
    gender: Optional[str] = None,
):
    """
    Debug endpoint for local testing (no upload).
    """
    if not os.path.exists(image_path):
        return JSONResponse(
            status_code=400,
            content={"error": "Image path does not exist"},
        )

    results = similar_items(
        image_path=image_path,
        n=n,
        subcategory=subcategory,
        gender=gender,
    )

    return {"results": results}
