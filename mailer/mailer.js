require('dotenv').config();
const nodemailer = require('nodemailer');
const fs = require('fs');
const path = require('path');

const PENDING_EMAILS_FILE = path.join(__dirname, '..', 'outputs', 'pending_emails.json');

const transporter = nodemailer.createTransport({
    service: 'gmail',  // Standard service, could be adjusted for Outlook/Yahoo etc.
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
    }
});

async function sendBulkEmails() {
    if (!fs.existsSync(PENDING_EMAILS_FILE)) {
        console.log("[INFO] No pending emails found.");
        process.exit(0);
    }

    const pendingData = JSON.parse(fs.readFileSync(PENDING_EMAILS_FILE, 'utf8'));
    let successCount = 0;
    
    console.log(`[INFO] Attempting to send ${pendingData.length} emails...`);

    for (const record of pendingData) {
        if (!record.email) continue;
        
        const mailOptions = {
            from: process.env.EMAIL_USER,
            to: record.email,
            subject: '⚠️ Traffic Violation Notice - Immediate Action Required',
            html: `
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <div style="background-color: #d73a49; padding: 20px; color: white; text-align: center;">
                        <h2 style="margin: 0;">Traffic Violation Notice</h2>
                    </div>
                    <div style="padding: 30px; background-color: #ffffff; color: #24292e;">
                        <p style="font-size: 16px;">Dear <strong>${record.owner_name}</strong>,</p>
                        <p style="font-size: 16px;">A traffic violation has been officially recorded for your registered vehicle: 
                            <strong style="background-color: #f6f8fa; padding: 4px 8px; border: 1px solid #d1d5da; border-radius: 4px;">${record.plate}</strong>
                        </p>
                        
                        <div style="background-color: #f1f8ff; border-left: 4px solid #0366d6; padding: 15px; margin: 25px 0;">
                            <ul style="list-style-type: none; padding: 0; margin: 0;">
                                <li style="margin-bottom: 8px;"><strong>Date & Time:</strong> ${record.timestamp}</li>
                                <li style="margin-bottom: 8px;"><strong>Violation Type:</strong> <span style="color: #d73a49; font-weight: bold;">${record.violation_type}</span></li>
                                <li><strong>Confidence Match:</strong> ${(record.confidence * 100).toFixed(1)}%</li>
                            </ul>
                        </div>
                        
                        <p style="font-size: 14px; color: #586069;">This is an automated system generated message. Please review the portal to view evidence pictures or contest this fine.</p>
                        
                        <div style="text-align: center; margin-top: 30px;">
                            <a href="#" style="background-color: #0366d6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: bold;">View Details / Pay Fine</a>
                        </div>
                    </div>
                    <div style="background-color: #f6f8fa; padding: 15px; text-align: center; font-size: 12px; color: #6a737d;">
                        &copy; 2026 Traffic Management Authority. All rights reserved.
                    </div>
                </div>
            `
        };

        try {
            await transporter.sendMail(mailOptions);
            console.log(`[SUCCESS] Sent to: ${record.email} (Plate: ${record.plate})`);
            successCount++;
        } catch (error) {
            console.error(`[ERROR] Failed to send to ${record.email}: ${error.message}`);
        }
    }
    
    console.log(`\n[COMPLETE] Successfully sent ${successCount} emails.`);
    
    // Clear out the file after finishing
    fs.writeFileSync(PENDING_EMAILS_FILE, JSON.stringify([]));
}

// Ensure the process exits nicely
sendBulkEmails().then(() => process.exit(0)).catch(e => {
    console.error(e);
    process.exit(1);
});
