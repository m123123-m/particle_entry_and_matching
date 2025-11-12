# Deployment Guide for Particle Entry and Matching Pipeline

## Quick Deploy to Vercel (Recommended)

### Prerequisites
```bash
npm install -g vercel
```

### Deploy Steps

1. **Login to Vercel:**
```bash
cd particle_entry_and_matching
vercel login
```

2. **Deploy:**
```bash
vercel
```

3. **For Production:**
```bash
vercel --prod
```

### Alternative: Deploy via GitHub

1. Push your code to GitHub (already done)
2. Go to https://vercel.com
3. Import your repository: `m123123-m/particle_entry_and_matching`
4. Vercel will auto-detect the Flask app
5. Click Deploy

## Important Notes

- **File Storage**: Vercel has ephemeral storage. For production, consider:
  - Using Vercel Blob Storage for uploaded files
  - Using external storage (S3, etc.) for data files
  - Or use a different platform like Render.com that has persistent storage

- **Large Simulations**: For simulations with >1M particles, consider:
  - Using background job processing
  - Increasing timeout limits
  - Using a platform with more compute resources

## Other Deployment Options

### Render.com (Recommended for File Storage)

1. Go to https://render.com
2. Create new Web Service
3. Connect GitHub repository: `m123123-m/particle_entry_and_matching`
4. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn web_app:app`
   - Environment: Python 3
   - Disk: Persistent (for data files)
5. Deploy

### Heroku

1. Install Heroku CLI
2. Create `Procfile`:
```
web: gunicorn web_app:app
```

3. Add buildpacks:
```bash
heroku buildpacks:add heroku/python
```

4. Deploy:
```bash
heroku create particle-entry-matching
git push heroku main
```

## Environment Variables

No environment variables required for basic operation.

## Post-Deployment

After deployment, your app will be available at:
- Vercel: `https://particle-entry-matching-xxxxx.vercel.app`
- Render: `https://particle-entry-matching.onrender.com`
- Heroku: `https://particle-entry-matching.herokuapp.com`

