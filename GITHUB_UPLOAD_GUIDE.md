# Publish This Project on GitHub

## 1. Create an empty repository

On GitHub, create a repository named `foundation-shade-recommender`.

Because this project already contains a README and `.gitignore`, leave the
repository initialization options unchecked.

## 2. Upload the project

Unzip the delivered project. On the empty GitHub repository page, choose
**uploading an existing file**, then drag all files and folders from inside
`foundation-shade-recommender` into the upload area.

Use this commit message:

```text
Add refactored foundation shade recommender
```

For ongoing work, GitHub Desktop or command-line Git will preserve version
history more conveniently than repeatedly uploading files in the browser.

## 3. Connect the notebooks to your repository

In every notebook's setup cell, replace:

```python
REPOSITORY_URL = "https://github.com/YOUR_USERNAME/foundation-shade-recommender.git"
```

with your real GitHub repository URL. Run the notebook from the beginning and
save the updated copy back to GitHub.

## 4. Before sharing the repository

- Run each notebook from top to bottom.
- Keep useful charts and summary tables, but clear noisy debugging output.
- Never commit a selfie, API key, password, large dataset, downloaded model, or
  generated FairFace images.
- Update the README's evaluation numbers only after reproducing them with the
  cleaned evaluation notebook.
- Add a short repository description such as: `Selfie-based foundation shade
  matching with MediaPipe, CIELAB, CIEDE2000, and K-Means.`
