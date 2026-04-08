import mysql.connector
from datetime import datetime

class DatabaseLogger:
    def __init__(self, config):
        self.config = config['database']
        self.conn = mysql.connector.connect(
            host=self.config['host'],
            user=self.config['user'],
            password=self.config['password'],
            database=self.config['database']
        )
        self.cursor = self.conn.cursor()
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                violation_id INT PRIMARY KEY AUTO_INCREMENT,
                timestamp DATETIME NOT NULL,
                camera_id VARCHAR(50),
                license_plate VARCHAR(20),
                confidence_score FLOAT,
                violation_type VARCHAR(50),
                snapshot_path VARCHAR(255),
                cropped_plate_path VARCHAR(255),
                status ENUM('pending', 'verified', 'dismissed'),
                INDEX idx_timestamp (timestamp),
                INDEX idx_plate (license_plate)
            )
        ''')
        self.conn.commit()

    def log_violation(self, camera_id, license_plate, confidence, violation_type, snapshot_path, cropped_plate_path):
        timestamp = datetime.now()
        sql = '''
            INSERT INTO violations (timestamp, camera_id, license_plate, confidence_score, violation_type, snapshot_path, cropped_plate_path, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        '''
        values = (timestamp, camera_id, license_plate, confidence, violation_type, snapshot_path, cropped_plate_path, 'pending')
        self.cursor.execute(sql, values)
        self.conn.commit()
