# 🚦 IntelliTraffic System - Traffic Violation Detection (YOLOv9 + EasyOCR)

An intelligent real-time traffic violation detection system using **YOLOv9** for motorcycle & helmet detection, **EasyOCR** for license plate recognition, and **Streamlit** for a modern web interface. Supports video uploads, live webcam feeds, RTSP streams, and bulk email notifications via Gmail.

## ✨ Features

- 🎥 **Real-time Detection**: Process MP4, webcam feeds, and RTSP streams
- 🏍️ **Motorcycle Detection**: Detect motorcycles and non-helmet riders using YOLOv9
- 📝 **License Plate OCR**: Indian license plate recognition with EasyOCR
- 💾 **Cloud Database**: Firebase Firestore integration for violation logging
- 📧 **Bulk Email System**: NodeMailer integration to send violation notices to vehicle owners
- 📊 **SQL Dashboard**: View detected violations with timestamps and confidence scores

## 📋 Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **GPU Support** (NVIDIA RTX 3060/4060+) - strongly recommended for EasyOCR
- **Node.js 18+** - required for email sending functionality
- **Gmail Account** - with App Password enabled for SMTP

## 🚀 Setup Instructions (Windows)

### Step 1: Clone & Install Python Dependencies

```powershell
cd "C:\Your\Project\Path\Traffic_Violation_Detection"
pip install -r requirements.txt
```

### Step 2: Setup Firebase Credentials

1. Download your Firebase service account JSON from Firebase Console
2. Rename it to `firebase_credentials.json`
3. Place it in the project root directory

⚠️ **IMPORTANT**: The `.gitignore` file WILL ignore this file - never commit credentials!

### Step 3: Configure Email Settings

1. Get your **Gmail App Password**:
   - Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
   - Enable 2-Step Verification first
   - Generate an App Password
2. Create/Update `mailer/.env`:
   ```
   EMAIL_USER=your_email@gmail.com
   EMAIL_PASS=your_app_password_here
   ```

### Step 4: Setup Node.js for Email Sending

```powershell
cd mailer
npm install
cd ..
```

### Step 5: Configure Your Settings

Edit `config.yaml` and update:

- Firebase database settings
- YOLOv9 model path (default: `yolov9c.pt`)
- Detection thresholds and output directories

### Step 6: Create Your Owner Database

Prepare a CSV file (`owners.csv`) with your vehicle owner data:

```csv
plate,owner_name,email
MH12AB1235,Rajesh,user@gmail.com
DL01CD5678,Priya,priya@gmail.com
```

**Required columns**: `plate`, `owner_name`, `email`

## 🎯 Running the Application

### Start the Streamlit Web App

```powershell
streamlit run app.py
```

The application will open at: **http://localhost:8501**

### Using the Web Interface

#### 📹 Tab 1 - Video / Live Detection

1. Select source: Upload Video/Image, Webcam, or RTSP Stream
2. Upload your video or configure stream URL
3. Set violation confidence threshold (0.1 - 1.0, default: 0.85)
4. Click **"▶ Start Feed"** to begin detection
5. Violations detected in real-time appear with red bounding boxes
6. Click **"⏹ Stop Feed"** to end

#### 📊 Tab 2 - SQL Database Dashboard

- View all detected violations
- See license plates, violation types, timestamps
- Track email sending status
- Real-time metrics (total records, recognized plates)

#### 📧 Tab 3 - Bulk Email System

1. Upload your owner CSV file
2. Click **"Verify matches & Send Bulk Emails 🚀"**
3. System automatically sends emails to all CSV owners:
   - ⚠️ **Red Alert Email** - if violation found for their plate
   - 📧 **Info Email** - if no violation (general notification)
4. Check **PowerShell Terminal** for real-time email logs

## 📁 Project Structure

```
Traffic_Violation_Detection/
├── app.py                          # Streamlit main application
├── main.py                         # Legacy CLI runner
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── yolov9c.pt                      # YOLOv9 model weights
├── firebase_credentials.json        # Firebase credentials (gitignored)
├── sample_owners.csv               # Sample vehicle owner database
│
├── src/
│   ├── detection.py               # YOLOv9 detection module
│   ├── ocr.py                     # License plate OCR
│   ├── violation_logic.py         # Violation checking logic
│   ├── cloud_db.py                # Firebase Firestore integration
│   └── db.py                      # MySQL database module
│
├── mailer/
│   ├── mailer.js                  # Node.js email sender
│   ├── package.json               # Node.js dependencies
│   └── .env                       # Gmail credentials (gitignored)
│
├── outputs/
│   ├── violations/                # Detected violation images
│   └── temp/                      # Temporary video files
│
└── .gitignore                     # Protects secrets & sensitive files
```

## ⚙️ Troubleshooting

### Issue: "Node.js is not installed"

**Solution**: Install Node.js from [nodejs.org](https://nodejs.org) (LTS version). Restart PowerShell after installation.

### Issue: Firebase credentials not found

**Solution**: Ensure `firebase_credentials.json` is in the project root and properly formatted.

### Issue: Gmail email not sending

**Solution**:

- Verify `.env` file exists in `mailer/` folder
- Check that EMAIL_USER and EMAIL_PASS are correct
- Ensure 2-Step Verification is enabled on your Gmail account
- Verify App Password was generated (not regular Gmail password)

### Issue: YOLOv9 model loading fails

**Solution**: Ensure `yolov9c.pt` is in the project root. Update `config.yaml` with correct path.

### Issue: "No matched plates found" warning but emails still sent

**This is expected behavior**. System sends to all CSV owners regardless of plate matches.

## 🔒 Security Notes

- ✅ `firebase_credentials.json` is in `.gitignore` - won't be committed
- ✅ `mailer/.env` is in `.gitignore` - Gmail credentials are protected
- ⚠️ Never commit these credential files to GitHub
- ⚠️ Use App Passwords, not your real Google password

## 📚 Dependencies

See `requirements.txt` for complete list. Key packages:

- `streamlit` - Web UI framework
- `torch`, `torchvision` - Deep learning
- `ultralytics` - YOLOv9 implementation
- `easyocr` - License plate OCR
- `firebase-admin` - Cloud database
- `opencv-python` - Video processing
- `pandas`, `numpy` - Data manipulation

## 🎓 Data Flow

```
Video Input → YOLOv9 Detection → OCR (License Plate) → Violation Check →
Firebase Logging → SQL Dashboard → Bulk Email (NodeMailer) → Gmail
```

## 📞 Support

For issues or questions about setup and usage, check the error messages in the Streamlit UI and PowerShell terminal logs.
