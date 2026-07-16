# Deploy to Vercel

## Goal

Deploy the current Gen-Aperture application to Vercel and return the resulting deployment URL.

## Non-goals

- Do not change application behavior unless deployment verification exposes a required packaging/configuration issue.
- Do not rotate, print, or copy secret values from local `.env` files.
- Do not add or upgrade dependencies as part of deployment.

## Constraints

- Use the managed Shutterstock agentic baseline for planning and verification.
- Keep diffs small and reviewable.
- Do not bypass Vercel, build, pre-commit, or CI checks.
- Use the already-installed Vercel CLI when possible.
- Avoid local package installation or dependency resolution unless explicitly required and approved.
- Treat production deployment and remote environment mutation as infrastructure-sensitive actions.

## Acceptance criteria

- The Vercel CLI is authenticated or the blocker is clearly reported.
- The frontend build succeeds locally from existing dependencies.
- The Vercel deployment command completes successfully.
- Server-side environment variables from `backend/.env` are copied to Vercel without exposing secret values.
- The final handoff includes deployment URL, commands run, what each proved, changed files, and remaining uncertainty.

## Approach

1. Inspect existing Vercel, frontend, and backend deployment configuration.
2. Run targeted local checks without dependency installation.
3. Deploy through the existing Vercel CLI.
4. Verify the returned deployment URL and basic HTTP health when available.
5. Copy required server-side environment variables from `backend/.env` into Vercel Preview and Production environments.

## Files / areas affected

- `docs/plans/deploy-vercel.md`
- Vercel project metadata may be created under `.vercel/` by the CLI.
- No application source changes are expected unless verification reveals a deployment blocker.

## Verification plan

- Run `npm --prefix frontend run build` to exercise the Vite frontend build using existing `node_modules`.
- Run a Vercel deployment command and capture the generated URL.
- Probe the deployed URL and health endpoint if deployment succeeds.
- Confirm Vercel lists the expected environment variable names without printing values.

## Test plan

- Before/proof: inspect repository deployment configuration and confirm Vercel CLI authentication.
- Happy path: local frontend build succeeds and Vercel returns a deployment URL.
- Sad path: if Vercel requires project linking, missing environment variables, or interactive credential input, stop and report the exact blocker.
- After/proof: verify the deployment URL responds over HTTP.
- Secret-copy proof: compare variable names from local `backend/.env` with Vercel `env ls` output; do not print secret values.

## Monitoring plan

- Use Vercel CLI output and deployment inspection as immediate post-deploy evidence.
- Surface any runtime health degradation, especially missing server-side API credentials or OpenSearch connectivity.

## Risks / open questions

- The backend depends on runtime secrets such as `NVIDIA_API_KEY`, `OPENAI_API_KEY`, and OpenSearch credentials; deployment may succeed while API health is degraded if they are not configured in Vercel.
- `frontend/package.json` declares Yarn while the root build script uses npm; the deploy may need Vercel project settings if remote install/build behavior differs from local checks.
- First-time Vercel project linking may create local `.vercel/` metadata.

## Status

- Preview deployment ready:
  - Deployment URL: `https://gen-aperture-boen8surk-rajanimaski2486s-projects.vercel.app`
  - Inspect URL: `https://vercel.com/rajanimaski2486s-projects/gen-aperture/GAMtTrfrZc2EEgEzNPnAQ8rASCiD`
- HTTP probes currently redirect to Vercel SSO because deployment protection is enabled for this project.
- Copied the six `backend/.env` values into Vercel as encrypted Preview and Production environment variables.
- Redeployed Preview after copying environment variables.
