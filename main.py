"""
Predictive Email Campaign Script

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Email and Database
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import json

class PredictiveEmailCampaign:
    def __init__(self, config_file='config.json'):
        """
        Initialize the predictive email campaign system
        """
        self.config = self.load_config(config_file)
        self.initialize_database()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            'database_path': 'email_campaigns.db',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'test_mode': True,
            'min_open_rate_for_sending': 0.1,
            'min_click_rate_for_sending': 0.02,
            'optimal_sending_hours': [9, 10, 14, 15, 16],
            'features': [
                'hour_of_day', 'day_of_week', 'subject_length',
                'body_length', 'has_links', 'has_images',
                'previous_opens', 'previous_clicks', 'days_since_last_contact',
                'user_engagement_score', 'timezone_offset'
            ]
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file not found, using default configuration")
            
        return default_config
    
    def initialize_database(self):
        """Initialize SQLite database for storing campaign data"""
        self.conn = sqlite3.connect(self.config['database_path'])
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_name TEXT,
                subject TEXT,
                body TEXT,
                target_segment TEXT,
                sent_date TIMESTAMP,
                total_sent INTEGER,
                total_opens INTEGER,
                total_clicks INTEGER,
                conversion_rate REAL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipients (
                recipient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                signup_date TIMESTAMP,
                location TEXT,
                timezone TEXT,
                total_opens INTEGER DEFAULT 0,
                total_clicks INTEGER DEFAULT 0,
                total_purchases INTEGER DEFAULT 0,
                engagement_score REAL DEFAULT 0,
                last_contact_date TIMESTAMP,
                segment TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_id INTEGER,
                campaign_id INTEGER,
                event_type TEXT,
                event_time TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (recipient_id) REFERENCES recipients (recipient_id),
                FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
            )
        ''')
        
        self.conn.commit()
    
    def generate_sample_data(self, num_recipients=1000, num_campaigns=50):
        """Generate sample data for testing"""
        print("Generating sample data...")
        
        # Generate sample recipients
        locations = ['US', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan']
        timezones = ['EST', 'PST', 'GMT', 'CET', 'JST', 'AEST']
        
        for i in range(num_recipients):
            email = f'user{i}@example.com'
            signup_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            location = np.random.choice(locations)
            timezone = np.random.choice(timezones)
            total_opens = np.random.randint(0, 100)
            total_clicks = np.random.randint(0, total_opens)
            total_purchases = np.random.randint(0, total_clicks)
            engagement_score = np.random.uniform(0, 1)
            last_contact_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO recipients 
                (email, signup_date, location, timezone, total_opens, total_clicks, 
                 total_purchases, engagement_score, last_contact_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (email, signup_date, location, timezone, total_opens, total_clicks,
                  total_purchases, engagement_score, last_contact_date))
        
        # Generate sample campaigns
        subjects = [
            "Special Offer Inside!", "Your Weekly Update", "Limited Time Discount",
            "New Features Available", "Important Announcement", "Personalized Recommendations"
        ]
        
        for i in range(num_campaigns):
            campaign_name = f"Campaign_{i+1}"
            subject = np.random.choice(subjects) + f" {np.random.randint(1, 100)}"
            body = f"This is the body of campaign {i+1}. Check out our amazing offers!"
            target_segment = np.random.choice(['all', 'active', 'inactive', 'high_value'])
            sent_date = datetime.now() - timedelta(days=np.random.randint(1, 60))
            total_sent = np.random.randint(100, 1000)
            total_opens = np.random.randint(int(total_sent * 0.1), int(total_sent * 0.5))
            total_clicks = np.random.randint(int(total_opens * 0.1), int(total_opens * 0.3))
            conversion_rate = np.random.uniform(0.01, 0.05)
            
            self.cursor.execute('''
                INSERT INTO campaigns 
                (campaign_name, subject, body, target_segment, sent_date, total_sent,
                 total_opens, total_clicks, conversion_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (campaign_name, subject, body, target_segment, sent_date, total_sent,
                  total_opens, total_clicks, conversion_rate))
            
            campaign_id = self.cursor.lastrowid
            
            # Generate sample email events
            recipients = list(range(1, min(num_recipients, total_sent) + 1))
            np.random.shuffle(recipients)
            
            for recipient_id in recipients[:total_sent]:
                # Open event
                if np.random.random() < 0.3:  # 30% open rate
                    open_time = sent_date + timedelta(hours=np.random.randint(1, 72))
                    self.cursor.execute('''
                        INSERT INTO email_events (recipient_id, campaign_id, event_type, event_time)
                        VALUES (?, ?, ?, ?)
                    ''', (recipient_id, campaign_id, 'open', open_time))
                    
                    # Click event (only if opened)
                    if np.random.random() < 0.2:  # 20% click-through rate
                        click_time = open_time + timedelta(minutes=np.random.randint(1, 60))
                        self.cursor.execute('''
                            INSERT INTO email_events (recipient_id, campaign_id, event_type, event_time)
                            VALUES (?, ?, ?, ?)
                        ''', (recipient_id, campaign_id, 'click', click_time))
        
        self.conn.commit()
        print(f"Generated {num_recipients} recipients and {num_campaigns} campaigns")
    
    def prepare_training_data(self):
        """Prepare training data from historical campaign data"""
        print("Preparing training data...")
        
        # Query to get features and labels
        query = '''
        SELECT 
            strftime('%H', e.event_time) as hour_of_day,
            strftime('%w', e.event_time) as day_of_week,
            LENGTH(c.subject) as subject_length,
            LENGTH(c.body) as body_length,
            CASE WHEN c.body LIKE '%http%' THEN 1 ELSE 0 END as has_links,
            CASE WHEN c.body LIKE '%<img%' THEN 1 ELSE 0 END as has_images,
            r.total_opens as previous_opens,
            r.total_clicks as previous_clicks,
            julianday(e.event_time) - julianday(r.last_contact_date) as days_since_last_contact,
            r.engagement_score,
            CASE 
                WHEN r.timezone = 'EST' THEN -5
                WHEN r.timezone = 'PST' THEN -8
                WHEN r.timezone = 'GMT' THEN 0
                WHEN r.timezone = 'CET' THEN 1
                WHEN r.timezone = 'JST' THEN 9
                WHEN r.timezone = 'AEST' THEN 10
                ELSE 0 
            END as timezone_offset,
            CASE WHEN e.event_type = 'open' THEN 1 ELSE 0 END as opened,
            CASE WHEN e.event_type = 'click' THEN 1 ELSE 0 END as clicked
        FROM email_events e
        JOIN campaigns c ON e.campaign_id = c.campaign_id
        JOIN recipients r ON e.recipient_id = r.recipient_id
        WHERE e.event_type IN ('open', 'click', 'sent')
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        # Convert columns to appropriate types
        df['hour_of_day'] = pd.to_numeric(df['hour_of_day'])
        df['day_of_week'] = pd.to_numeric(df['day_of_week'])
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    
    def train_prediction_models(self):
        """Train machine learning models for different predictions"""
        print("Training prediction models...")
        
        df = self.prepare_training_data()
        
        if len(df) < 100:
            print("Insufficient data for training. Generating sample data...")
            self.generate_sample_data()
            df = self.prepare_training_data()
        
        # Features and labels
        X = df[self.config['features']]
        y_open = df['opened']
        y_click = df['clicked']
        
        # Split data
        X_train, X_test, y_open_train, y_open_test = train_test_split(
            X, y_open, test_size=0.2, random_state=42
        )
        
        _, _, y_click_train, y_click_test = train_test_split(
            X, y_click, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train open prediction model
        print("Training open prediction model...")
        open_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        open_model.fit(X_train_scaled, y_open_train)
        self.models['open_prediction'] = open_model
        
        # Train click prediction model
        print("Training click prediction model...")
        click_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        click_model.fit(X_train_scaled, y_click_train)
        self.models['click_prediction'] = click_model
        
        # Evaluate models
        open_pred = open_model.predict(X_test_scaled)
        click_pred = click_model.predict(X_test_scaled)
        
        print("\n=== Model Performance ===")
        print("Open Prediction Accuracy:", accuracy_score(y_open_test, open_pred))
        print("Click Prediction Accuracy:", accuracy_score(y_click_test, click_pred))
        
        return open_model, click_model
    
    def predict_optimal_time(self, recipient_data, campaign_data):
        """Predict optimal sending time for a recipient"""
        # Prepare feature vector
        features = {}
        
        # Try different sending hours
        best_hour = None
        best_score = -1
        
        for hour in range(24):
            # Create feature vector for this hour
            feature_vector = [
                hour,  # hour_of_day
                datetime.now().weekday(),  # day_of_week
                len(campaign_data.get('subject', '')),  # subject_length
                len(campaign_data.get('body', '')),  # body_length
                1 if 'http' in campaign_data.get('body', '') else 0,  # has_links
                1 if '<img' in campaign_data.get('body', '') else 0,  # has_images
                recipient_data.get('total_opens', 0),  # previous_opens
                recipient_data.get('total_clicks', 0),  # previous_clicks
                (datetime.now() - pd.to_datetime(recipient_data.get('last_contact_date', datetime.now()))).days,  # days_since_last_contact
                recipient_data.get('engagement_score', 0),  # user_engagement_score
                recipient_data.get('timezone_offset', 0)  # timezone_offset
            ]
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Predict open probability
            open_prob = self.models['open_prediction'].predict_proba(feature_vector_scaled)[0][1]
            click_prob = self.models['click_prediction'].predict_proba(feature_vector_scaled)[0][1]
            
            # Combined score (weighted)
            score = (0.7 * open_prob) + (0.3 * click_prob)
            
            if score > best_score:
                best_score = score
                best_hour = hour
        
        return {
            'optimal_hour': best_hour,
            'open_probability': self.models['open_prediction'].predict_proba(
                self.scaler.transform([[
                    best_hour,
                    datetime.now().weekday(),
                    len(campaign_data.get('subject', '')),
                    len(campaign_data.get('body', '')),
                    1 if 'http' in campaign_data.get('body', '') else 0,
                    1 if '<img' in campaign_data.get('body', '') else 0,
                    recipient_data.get('total_opens', 0),
                    recipient_data.get('total_clicks', 0),
                    (datetime.now() - pd.to_datetime(recipient_data.get('last_contact_date', datetime.now()))).days,
                    recipient_data.get('engagement_score', 0),
                    recipient_data.get('timezone_offset', 0)
                ]])
            )[0][1],
            'click_probability': self.models['click_prediction'].predict_proba(
                self.scaler.transform([[
                    best_hour,
                    datetime.now().weekday(),
                    len(campaign_data.get('subject', '')),
                    len(campaign_data.get('body', '')),
                    1 if 'http' in campaign_data.get('body', '') else 0,
                    1 if '<img' in campaign_data.get('body', '') else 0,
                    recipient_data.get('total_opens', 0),
                    recipient_data.get('total_clicks', 0),
                    (datetime.now() - pd.to_datetime(recipient_data.get('last_contact_date', datetime.now()))).days,
                    recipient_data.get('engagement_score', 0),
                    recipient_data.get('timezone_offset', 0)
                ]])
            )[0][1]
        }
    
    def segment_recipients(self, n_clusters=4):
        """Segment recipients using clustering"""
        print("Segmenting recipients...")
        
        # Get recipient data
        query = '''
        SELECT 
            recipient_id,
            total_opens,
            total_clicks,
            total_purchases,
            engagement_score,
            julianday('now') - julianday(last_contact_date) as days_since_last_contact
        FROM recipients
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) < n_clusters:
            print(f"Not enough recipients for {n_clusters} clusters")
            return
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[['total_opens', 'total_clicks', 
                                               'total_purchases', 'engagement_score',
                                               'days_since_last_contact']])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        df['segment'] = clusters
        
        # Update segments in database
        for _, row in df.iterrows():
            segment_name = f"segment_{row['segment']}"
            self.cursor.execute(
                "UPDATE recipients SET segment = ? WHERE recipient_id = ?",
                (segment_name, row['recipient_id'])
            )
        
        self.conn.commit()
        
        # Analyze segments
        segment_analysis = df.groupby('segment').agg({
            'total_opens': 'mean',
            'total_clicks': 'mean',
            'total_purchases': 'mean',
            'engagement_score': 'mean',
            'days_since_last_contact': 'mean',
            'recipient_id': 'count'
        }).round(2)
        
        segment_analysis.columns = ['avg_opens', 'avg_clicks', 'avg_purchases', 
                                    'avg_engagement', 'avg_days_since_contact', 'count']
        
        print("\n=== Recipient Segments ===")
        print(segment_analysis)
        
        return segment_analysis
    
    def send_predictive_campaign(self, campaign_name, subject, body, target_segment=None):
        """Send an email campaign using predictive insights"""
        print(f"\nSending campaign: {campaign_name}")
        
        # Get recipients
        if target_segment:
            query = "SELECT * FROM recipients WHERE segment = ?"
            recipients_df = pd.read_sql_query(query, self.conn, params=(target_segment,))
        else:
            query = "SELECT * FROM recipients"
            recipients_df = pd.read_sql_query(query, self.conn)
        
        if len(recipients_df) == 0:
            print("No recipients found!")
            return
        
        campaign_data = {
            'subject': subject,
            'body': body
        }
        
        results = []
        
        for _, recipient in recipients_df.iterrows():
            # Predict optimal sending time
            prediction = self.predict_optimal_time(recipient.to_dict(), campaign_data)
            
            # Check if recipient is likely to engage
            if (prediction['open_probability'] > self.config['min_open_rate_for_sending'] and
                prediction['click_probability'] > self.config['min_click_rate_for_sending']):
                
                # In test mode, just simulate sending
                if self.config['test_mode']:
                    print(f"Would send to {recipient['email']} at hour {prediction['optimal_hour']}: "
                          f"Open prob: {prediction['open_probability']:.2%}, "
                          f"Click prob: {prediction['click_probability']:.2%}")
                    
                    # Simulate response based on probabilities
                    if np.random.random() < prediction['open_probability']:
                        event_type = 'open'
                        if np.random.random() < prediction['click_probability']:
                            event_type = 'click'
                    else:
                        event_type = 'sent'
                    
                    # Record simulated event
                    self.cursor.execute('''
                        INSERT INTO email_events 
                        (recipient_id, campaign_id, event_type, event_time)
                        VALUES (?, ?, ?, ?)
                    ''', (recipient['recipient_id'], 1, event_type, datetime.now()))
                
                results.append({
                    'email': recipient['email'],
                    'optimal_hour': prediction['optimal_hour'],
                    'open_probability': prediction['open_probability'],
                    'click_probability': prediction['click_probability'],
                    'sent': True
                })
            else:
                results.append({
                    'email': recipient['email'],
                    'optimal_hour': prediction['optimal_hour'],
                    'open_probability': prediction['open_probability'],
                    'click_probability': prediction['click_probability'],
                    'sent': False,
                    'reason': 'Low engagement probability'
                })
        
        self.conn.commit()
        
        # Save campaign to database
        total_sent = sum(1 for r in results if r['sent'])
        self.cursor.execute('''
            INSERT INTO campaigns 
            (campaign_name, subject, body, target_segment, sent_date, total_sent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (campaign_name, subject, body, target_segment, datetime.now(), total_sent))
        
        self.conn.commit()
        
        # Print summary
        sent_count = sum(1 for r in results if r['sent'])
        print(f"\n=== Campaign Summary ===")
        print(f"Total recipients: {len(recipients_df)}")
        print(f"Emails sent: {sent_count}")
        print(f"Filtered out: {len(recipients_df) - sent_count}")
        
        if sent_count > 0:
            avg_open_prob = np.mean([r['open_probability'] for r in results if r['sent']])
            avg_click_prob = np.mean([r['click_probability'] for r in results if r['sent']])
            print(f"Average open probability: {avg_open_prob:.2%}")
            print(f"Average click probability: {avg_click_prob:.2%}")
        
        return results
    
    def analyze_campaign_performance(self):
        """Analyze performance of past campaigns"""
        print("\n=== Campaign Performance Analysis ===")
        
        query = '''
        SELECT 
            campaign_name,
            target_segment,
            total_sent,
            total_opens,
            total_clicks,
            conversion_rate,
            sent_date
        FROM campaigns
        ORDER BY sent_date DESC
        LIMIT 10
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) > 0:
            df['open_rate'] = (df['total_opens'] / df['total_sent']).round(3)
            df['click_rate'] = (df['total_clicks'] / df['total_opens']).round(3)
            df['ctr'] = (df['total_clicks'] / df['total_sent']).round(3)
            
            print(df[['campaign_name', 'target_segment', 'total_sent', 
                      'open_rate', 'click_rate', 'ctr']].to_string(index=False))
            
            # Visualize performance
            self.visualize_performance(df)
        else:
            print("No campaign data available")
        
        return df
    
    def visualize_performance(self, df):
        """Create visualizations of campaign performance"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Open Rate by Segment
        if 'target_segment' in df.columns and 'open_rate' in df.columns:
            segment_open_rate = df.groupby('target_segment')['open_rate'].mean()
            axes[0, 0].bar(segment_open_rate.index, segment_open_rate.values)
            axes[0, 0].set_title('Average Open Rate by Segment')
            axes[0, 0].set_ylabel('Open Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Click Rate by Segment
        if 'target_segment' in df.columns and 'click_rate' in df.columns:
            segment_click_rate = df.groupby('target_segment')['click_rate'].mean()
            axes[0, 1].bar(segment_click_rate.index, segment_click_rate.values)
            axes[0, 1].set_title('Average Click Rate by Segment')
            axes[0, 1].set_ylabel('Click Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Open Rate Over Time
        if 'sent_date' in df.columns and 'open_rate' in df.columns:
            df['sent_date'] = pd.to_datetime(df['sent_date'])
            df_sorted = df.sort_values('sent_date')
            axes[1, 0].plot(df_sorted['sent_date'], df_sorted['open_rate'], marker='o')
            axes[1, 0].set_title('Open Rate Trend')
            axes[1, 0].set_ylabel('Open Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Campaign Performance Scatter
        if 'total_sent' in df.columns and 'open_rate' in df.columns:
            axes[1, 1].scatter(df['total_sent'], df['open_rate'], alpha=0.6)
            axes[1, 1].set_title('Campaign Volume vs Open Rate')
            axes[1, 1].set_xlabel('Total Sent')
            axes[1, 1].set_ylabel('Open Rate')
        
        plt.tight_layout()
        plt.savefig('campaign_performance.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def get_recommendations(self):
        """Generate recommendations for improving campaigns"""
        print("\n=== Recommendations ===")
        
        # Analyze historical data for patterns
        query = '''
        SELECT 
            strftime('%H', event_time) as hour,
            COUNT(CASE WHEN event_type = 'open' THEN 1 END) as opens,
            COUNT(CASE WHEN event_type = 'click' THEN 1 END) as clicks,
            COUNT(*) as total_events
        FROM email_events
        GROUP BY strftime('%H', event_time)
        ORDER BY opens DESC
        LIMIT 5
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) > 0:
            df['open_rate'] = (df['opens'] / df['total_events']).round(3)
            df['click_rate'] = (df['clicks'] / df['opens']).round(3)
            
            print("Top performing sending hours:")
            for _, row in df.iterrows():
                print(f"  Hour {int(row['hour'])}: "
                      f"Open rate: {row['open_rate']:.1%}, "
                      f"Click rate: {row['click_rate']:.1%}")
        
        # Segment performance analysis
        segment_query = '''
        SELECT 
            r.segment,
            COUNT(CASE WHEN e.event_type = 'open' THEN 1 END) as opens,
            COUNT(CASE WHEN e.event_type = 'click' THEN 1 END) as clicks,
            COUNT(DISTINCT e.recipient_id) as recipients
        FROM recipients r
        LEFT JOIN email_events e ON r.recipient_id = e.recipient_id
        WHERE r.segment IS NOT NULL
        GROUP BY r.segment
        '''
        
        segment_df = pd.read_sql_query(segment_query, self.conn)
        
        if len(segment_df) > 0:
            segment_df['open_rate'] = (segment_df['opens'] / segment_df['recipients']).round(3)
            segment_df['click_rate'] = (segment_df['clicks'] / segment_df['opens']).round(3)
            
            print("\nSegment performance:")
            for _, row in segment_df.iterrows():
                print(f"  {row['segment']}: "
                      f"Open rate: {row['open_rate']:.1%}, "
                      f"Click rate: {row['click_rate']:.1%}")
    
    def run_demo(self):
        """Run a complete demo of the system"""
        print("=" * 50)
        print("PREDICTIVE EMAIL CAMPAIGN SYSTEM")
        print("=" * 50)
        
        # Step 1: Generate sample data if needed
        self.generate_sample_data(num_recipients=500, num_campaigns=20)
        
        # Step 2: Train models
        self.train_prediction_models()
        
        # Step 3: Segment recipients
        self.segment_recipients(n_clusters=4)
        
        # Step 4: Send predictive campaign
        campaign_results = self.send_predictive_campaign(
            campaign_name="Spring Sale 2024",
            subject="Exclusive 30% Discount for You!",
            body="Dear Customer,\n\nWe're excited to offer you an exclusive 30% discount on all products. "
                 "Visit our website to claim your offer!\n\nBest regards,\nThe Marketing Team",
            target_segment="segment_0"  # Most engaged segment
        )
        
        # Step 5: Analyze performance
        self.analyze_campaign_performance()
        
        # Step 6: Get recommendations
        self.get_recommendations()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("=" * 50)

# Main execution
if __name__ == "__main__":
    # Initialize the system
    campaign_system = PredictiveEmailCampaign()
    
    # Run demo
    campaign_system.run_demo()
    
    # Example of manual campaign sending
    # campaign_system.send_predictive_campaign(
    #     campaign_name="Test Campaign",
    #     subject="Test Subject",
    #     body="Test email body",
    #     target_segment="segment_0"
    # )