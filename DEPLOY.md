# Deployment Guide for XRD Plotter on Streamlit Cloud

This guide explains how to deploy the XRD Plotter application on Streamlit Cloud and addresses common issues, including slow deployment.

## Prerequisites

- A GitHub account connected to the repository
- The repository containing the XRD Plotter code
- Admin access to the repository

## Optimized Deployment Steps

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

2. Click the "New app" button in the dashboard.

3. Fill in the deployment details:
   - **Repository**: Choose your GitHub repository (e.g., `jrjfonseca/XRD_PLOTTER`)
   - **Branch**: Select the branch (usually `main`)
   - **Main file path**: Enter `streamlit_app.py` (important: use this filename instead of app.py)
   - **App URL**: Choose a custom URL or use the default one provided

4. Click "Deploy!" to start the deployment process.

5. Streamlit Cloud will automatically install the dependencies from `requirements.txt` and launch your app.

## Addressing Slow Deployment Issues

If your app is taking a long time to deploy, try these solutions:

1. **Use streamlit_app.py as the main file**: This is Streamlit Cloud's preferred filename.

2. **Optimize requirements.txt**:
   - Use version ranges (e.g., `streamlit>=1.24.0`) instead of exact versions
   - Remove unnecessary dependencies
   - Minimize large dependencies when possible

3. **Lazy loading of optional dependencies**:
   - The updated code uses `@st.cache_resource` to load scienceplots only when needed
   - This reduces startup time by not loading heavy libraries until requested

4. **Reduce image and asset sizes**:
   - Compress any images used in the app
   - Use web-optimized formats when possible

5. **Use smaller datasets for testing**:
   - Start with minimal data to ensure deployment works
   - Add larger datasets after confirming deployment success

## Checking Deployment Status

- The deployment process will be shown in real-time.
- Once complete, you'll see a URL where your app is hosted.
- The app will be publicly accessible at this URL.

## Monitoring and Debugging

Streamlit Cloud provides logs to help diagnose deployment issues:

1. From your app dashboard, click on the app you're deploying
2. Click on "Manage app" in the top-right corner
3. Select the "Logs" tab to view detailed deployment logs
4. Look for errors or warnings that might indicate package installation issues

## Updating the App

- Any changes pushed to the GitHub repository's main branch will automatically trigger a redeployment.
- You can manually redeploy from the Streamlit Cloud dashboard if needed.

## Advanced Troubleshooting

If your app still deploys slowly:

1. **Use smaller dependencies**: Consider alternatives to heavy libraries like scienceplots
2. **Implement caching**: Add `@st.cache_data` and `@st.cache_resource` to heavy functions
3. **Split the app**: Consider breaking large apps into multiple smaller apps
4. **Contact Streamlit Support**: If issues persist, contact Streamlit's support team

## Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Cloud Resource Limits](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/app-limits)
- [Streamlit Performance Optimization](https://docs.streamlit.io/library/api-reference/performance)
- [Streamlit Community](https://discuss.streamlit.io) 