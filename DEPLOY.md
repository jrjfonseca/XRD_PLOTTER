# Deployment Guide for XRD Plotter on Streamlit Cloud

This guide explains how to deploy the XRD Plotter application on Streamlit Cloud.

## Prerequisites

- A GitHub account connected to the repository
- The repository containing the XRD Plotter code
- Admin access to the repository

## Deployment Steps

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

2. Click the "New app" button in the dashboard.

3. Fill in the deployment details:
   - **Repository**: Choose your GitHub repository (e.g., `jrjfonseca/XRD_PLOTTER`)
   - **Branch**: Select the branch (usually `main`)
   - **Main file path**: Enter `app.py`
   - **App URL**: Choose a custom URL or use the default one provided

4. Click "Deploy!" to start the deployment process.

5. Streamlit Cloud will automatically install the dependencies from `requirements.txt` and launch your app.

## Checking Deployment Status

- The deployment process will be shown in real-time.
- Once complete, you'll see a URL where your app is hosted.
- The app will be publicly accessible at this URL.

## Updating the App

- Any changes pushed to the GitHub repository's main branch will automatically trigger a redeployment.
- You can manually redeploy from the Streamlit Cloud dashboard if needed.

## Troubleshooting

If you encounter issues during deployment:

1. Check the build logs in the Streamlit Cloud dashboard.
2. Verify that all dependencies are correctly listed in `requirements.txt`.
3. Ensure the app works locally before attempting deployment.
4. Check for any package version incompatibilities.

## Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community](https://discuss.streamlit.io) 