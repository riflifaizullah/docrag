# Deploying a public demo

DocRAG is built to run on your own machine, with a local language model and no sign-in. Putting it on the internet as it is would be a mistake. This page describes the one safe way to publish something: a **read-only demo** of the retrieval, with public sample documents and no language model.

## What the demo is

Start it with `docker compose -f docker-compose.demo.yml up --build` and open http://localhost:5080. In demo mode (`DOCRAG_DEMO=1`):

- **No language model.** A question returns the best matching passages with their document and page, or the usual "not found" when nothing relevant is retrieved. It shows retrieval, not written answers.
- **Read-only.** Re-indexing, flagging, exporting, chunk inspection and the monitor page return 403.
- **Rate limited.** At most 30 questions a minute across all visitors, then 429. This protects a small free host and is not a defence against a determined attacker.
- **Public documents only.** The backend has no sign-in. Anything indexed can be read by anyone who can reach it. Use the fictional sample documents or other public ones, never private files.

The demo needs about 2 GB of memory for the embedding and reranking models, which is more than most free tiers give. Check the host's limits before choosing one.

## Publishing the images

The workflow `.github/workflows/publish-images.yml` builds the backend and website images and publishes them to GitHub's container registry. It runs only when you push a version tag, never on an ordinary push:

```bash
git tag v1.0.0
git push origin v1.0.0
```

That creates `ghcr.io/<owner>/docrag-backend` and `ghcr.io/<owner>/docrag-website`. New packages start private. Make them public in the repository's Packages settings if you want others to pull them.

## Running the images on a host

Any host that runs two containers can serve the demo. For each, set:

| Container | Settings |
|---|---|
| backend | `DOCRAG_DEMO=1`, `DOCRAG_VLM=0`, `DOCRAG_ALLOWED_HOSTS` set to the backend's own host name (and `localhost`), documents mounted at `/data/corpus` |
| website | `DOCRAG_BACKEND_URL` pointing at the backend, port 5080 |

The website also asks the browser to fetch PDFs from `http://localhost:8000` for the preview panel, so the PDF preview will not work on a public host without a change. Answers and search do not need it.

## What has and has not been tested

- Demo mode has unit tests (rate limit, blocked endpoints, the passage-only answer in both languages).
- The demo Docker setup is started and queried in CI, and I ran it locally.
- **Nothing here has been deployed to a real host.** I have not created a cloud account or tested the image publishing workflow, since that needs a version tag and your account. Treat the host steps as a guide and expect to adjust them.
- The PDF preview limitation above is known and not fixed.
