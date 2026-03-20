"""
BSM Exhibition Backend - Flask Server (Vercel-ready version)
Handles API requests, file uploads, OCR processing, and Turso/SQLite database
Optimized for Vercel's serverless environment
"""
import os
import sqlite3
import libsql
import uuid
import json
import base64
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai

# Configuration
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    BASE_DIR = '/var/task'
# Use /tmp for Vercel (ephemeral filesystem)
if os.environ.get('VERCEL'):
    DATABASE_PATH = '/tmp/database.db'
else:
    DATABASE_PATH = os.path.join(BASE_DIR, 'database.db')

# Turso configuration
TURSO_DB_URL = os.environ.get('TURSO_DATABASE_URL')
TURSO_AUTH_TOKEN = os.environ.get('TURSO_AUTH_TOKEN')

# Track if Turso is available (updated after connection test)
TURSO_AVAILABLE = False
_turso_conn = None  # Reuse Turso connection across requests

class _NoCloseConnection:
    """Wrapper that ignores close() calls to reuse a single Turso connection."""
    def __init__(self, conn):
        self._conn = conn
    def __getattr__(self, name):
        return getattr(self._conn, name)
    def close(self):
        pass  # Don't close — reuse across requests

def get_db():
    """Get database connection - Turso for production, SQLite for local dev.
    Reuses a single connection to avoid per-request connection overhead.
    Returns a wrapped connection where close() is a no-op (safe to call everywhere).
    Uses HTTP endpoint for serverless compatibility (libsql:// websockets can be unreliable in Lambda)."""
    global _turso_conn
    if TURSO_DB_URL and TURSO_AUTH_TOKEN and TURSO_AVAILABLE:
        try:
            # Use HTTP URL for serverless compatibility — websockets are problematic in Lambda
            http_url = TURSO_DB_URL
            if http_url.startswith('libsql://'):
                http_url = http_url.replace('libsql://', 'https://')

            if _turso_conn is None:
                _turso_conn = libsql.connect(http_url, auth_token=TURSO_AUTH_TOKEN)
            return _NoCloseConnection(_turso_conn)
        except Exception as e:
            print(f"Turso connection failed: {e}, resetting connection")
            _turso_conn = None
    
    # Use local SQLite (development or fallback) — close() works normally here
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like row access for dict_from_row()
    return conn

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Flask app setup
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Database initialization
def init_database():
    """Initialize database with contacts and industries tables"""
    global TURSO_AVAILABLE
    
    # Test Turso connection first
    if TURSO_DB_URL and TURSO_AUTH_TOKEN:
        try:
            test_conn = libsql.connect(TURSO_DB_URL, auth_token=TURSO_AUTH_TOKEN)
            test_conn.execute('SELECT 1')
            del test_conn  # Don't call close() — let it drop
            TURSO_AVAILABLE = True
            print(f"✅ Turso connection verified: {TURSO_DB_URL}")
        except Exception as e:
            print(f"❌ Turso connection test failed: {type(e).__name__}: {e}")
            TURSO_AVAILABLE = False
    else:
        print(f"⚠️ Turso not configured: URL={'set' if TURSO_DB_URL else 'missing'}, TOKEN={'set' if TURSO_AUTH_TOKEN else 'missing'}")
        TURSO_AVAILABLE = False
    
    db = get_db()
    
    # Turso uses standard SQL, SQLite also supports these syntaxes
    db.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            company TEXT,
            designation TEXT,
            phone TEXT,
            email TEXT,
            website TEXT,
            industry TEXT,
            sales_person TEXT,
            notes TEXT DEFAULT '',
            images TEXT,
            created_at TEXT
        )
    ''')
    
    db.execute('''
        CREATE TABLE IF NOT EXISTS industries (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            is_default INTEGER DEFAULT 0
        )
    ''')
    
    db.execute('''
        CREATE TABLE IF NOT EXISTS sales_persons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            is_default INTEGER DEFAULT 0
        )
    ''')
    
    # Insert default industries if not exist
    default_industries = [
        ('it', 'IT', 1),
        ('finance', 'Finance', 1),
        ('healthcare', 'Healthcare', 1),
        ('manufacturing', 'Manufacturing', 1),
        ('retail', 'Retail', 1),
        ('realestate', 'Real Estate', 1),
        ('education', 'Education', 1),
        ('marketing', 'Marketing', 1),
        ('consulting', 'Consulting', 1),
        ('other', 'Other (Type Your Own)', 1)
    ]
    
    for industry in default_industries:
        db.execute('INSERT OR IGNORE INTO industries (id, name, is_default) VALUES (?, ?, ?)', industry)
    
    # Insert default sales persons if not exist
    default_sales_persons = [
        ('sp1', 'Self', 1),
        ('sp2', 'Tushar', 1),
        ('sp3', 'Ainesh', 1),
    ]
    
    for sp in default_sales_persons:
        db.execute('INSERT OR IGNORE INTO sales_persons (id, name, is_default) VALUES (?, ?, ?)', sp)
    
    # Add notes column to contacts if not exists
    try:
        db.execute('ALTER TABLE contacts ADD COLUMN notes TEXT DEFAULT ""')
    except:
        pass  # Column may already exist
    
    # Add index on created_at for fast ordering
    try:
        db.execute('CREATE INDEX IF NOT EXISTS idx_contacts_created ON contacts(created_at)')
    except:
        pass
    
    db.commit()
    db.close()
    db_type = "Turso" if TURSO_AVAILABLE else "SQLite"
    print(f"Database initialized ({db_type}) at: {TURSO_DB_URL if TURSO_AVAILABLE else DATABASE_PATH}")

# Database helpers
def get_db_connection():
    """Get database connection with proper row handling"""
    db = get_db()
    # For SQLite compatibility with dict access
    if not TURSO_AVAILABLE:
        db.row_factory = sqlite3.Row
    return db

def dict_from_row(row):
    """Convert Row to dict - handles both SQLite Row and Turso results"""
    if row is None:
        return None
    # SQLite Row objects have _asdict()
    if hasattr(row, '_asdict'):
        return row._asdict()
    # Turso rows might be tuples or have different structure
    if hasattr(row, 'keys'):
        try:
            return dict(row)
        except:
            pass
    # Fallback: try iterating as tuple with column names
    # This handles libsql results that come as tuples
    try:
        # Get column names from description if available
        if hasattr(row, '_description'):
            desc = row._description
            return {desc[i][0]: row[i] for i in range(len(desc))}
    except:
        pass
    # Last resort: try direct dict conversion
    try:
        return dict(row)
    except:
        return {'_raw': str(row)}

# GPT-4o Vision OCR function (synchronous for Vercel)
def extract_with_gpt4v(image_base64):
    """
    Extract contact info from business card image using GPT-4o Vision.
    Falls back to regex parsing if API fails.
    """
    if not OPENAI_API_KEY or not openai_client:
        raise Exception("OPENAI_API_KEY not configured")
    
    try:
        # Prepare the image for OpenAI
        if ',' in image_base64:
            # Remove data URL prefix if present
            image_base64 = image_base64.split(',')[1]
        
        # Call OpenAI GPT-4o with vision
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all contact information from this business card. Return ONLY valid JSON with these exact fields: fullName, company, designation, phone, email, website. Do not include any explanation or markdown formatting."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            extracted_data = json.loads(json_match.group())
            
            # Normalize field names
            return {
                'fullName': extracted_data.get('fullName', extracted_data.get('name', '')),
                'company': extracted_data.get('company', ''),
                'designation': extracted_data.get('designation', extracted_data.get('title', '')),
                'phone': extracted_data.get('phone', ''),
                'email': extracted_data.get('email', ''),
                'website': extracted_data.get('website', '')
            }
        else:
            raise Exception("Could not parse JSON from GPT-4o response")
            
    except Exception as e:
        print(f"GPT-4o Vision OCR failed: {e}")
        raise

def parse_ocr_text(text):
    """Parse text to extract contact information (fallback)"""
    import re
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    result = {
        'fullName': '',
        'company': '',
        'designation': '',
        'phone': '',
        'email': '',
        'website': ''
    }
    
    # Email pattern
    email_regex = r'[\w.-]+@[\w.-]+\.\w+'
    emails = re.findall(email_regex, text)
    if emails:
        result['email'] = emails[0]
    
    # Website pattern
    website_regex = r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|io|co|biz|info|me|us|uk|in|au|ca|xyz|tech|app|dev|live|online|site|web|store|shop)(?:/[\w\-./?%&=]*)?'
    websites = re.findall(website_regex, text, re.IGNORECASE)
    if websites:
        website = websites[0]
        if not website.startswith('http'):
            website = 'https://' + website
        result['website'] = website
    
    # Phone pattern
    phone_regex = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
    phones = re.findall(phone_regex, text)
    if phones:
        valid_phone = next((p for p in phones if len(re.sub(r'\D', '', p)) >= 8), None)
        if valid_phone:
            result['phone'] = valid_phone
    
    # Company keywords
    company_keywords = ['inc', 'corp', 'llc', 'ltd', 'co.', 'company', 'group', 'solutions', 
                        'services', 'technologies', 'tech', 'systems', 'enterprises', 'pvt', 'private']
    
    # Title keywords
    title_keywords = ['ceo', 'cto', 'cfo', 'director', 'manager', 'president', 'vp', 'head', 
                     'lead', 'engineer', 'developer', 'designer', 'consultant', 'analyst', 
                     'executive', 'officer', 'chief', 'senior', 'junior', 'associate', 
                     'coordinator', 'specialist', 'representative', 'sales', 'marketing', 
                     'account', 'partner', 'founder', 'owner']
    
    for line in lines:
        lower_line = line.lower()
        
        # Extract company
        if not result['company']:
            for keyword in company_keywords:
                if keyword in lower_line:
                    result['company'] = line
                    break
        
        # Extract designation
        if not result['designation']:
            for keyword in title_keywords:
                if keyword in lower_line:
                    result['designation'] = line
                    break
        
        # Extract name (simple heuristic)
        if not result['fullName']:
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$', line):
                if not any(k in lower_line for k in company_keywords + title_keywords):
                    if len(line.split()) <= 4:
                        result['fullName'] = line
    
    return result

# Synchronous OCR processing for Vercel
def process_ocr_sync(base64_data):
    """Process OCR synchronously - suitable for Vercel serverless"""
    try:
        if not OPENAI_API_KEY:
            return {
                'fullName': '',
                'company': '',
                'designation': '',
                'phone': '',
                'email': '',
                'website': '',
                '_warning': 'OPENAI_API_KEY not configured'
            }
        
        # Try GPT-4o Vision first
        try:
            extracted_data = extract_with_gpt4v(base64_data)
            return extracted_data
        except Exception as gpt_error:
            print(f"GPT-4o Vision failed: {gpt_error}")
            return {
                'fullName': '',
                'company': '',
                'designation': '',
                'phone': '',
                'email': '',
                'website': '',
                '_fallback_note': 'GPT-4o Vision failed. Install easyocr for fallback.'
            }
        
    except Exception as e:
        return {'error': str(e)}

# API Routes
@app.route('/')
def index():
    """Serve the main frontend"""
    static_dir = os.path.join(BASE_DIR, 'static')
    if not os.path.exists(static_dir):
        static_dir = BASE_DIR  # Fallback
    return send_from_directory(static_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    static_dir = os.path.join(BASE_DIR, 'static')
    if not os.path.exists(static_dir):
        static_dir = BASE_DIR  # Fallback
    return send_from_directory(static_dir, filename)

# Contacts API
@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    """Get all contacts"""
    db = get_db()
    try:
        # Turso: use db.execute() and fetchall()
        # Note: images excluded from list view for performance (12MB+ savings)
        # Fetch images individually per contact via /api/contacts/<id>/images
        result = db.execute('SELECT id, full_name, company, designation, phone, email, website, industry, notes, created_at FROM contacts ORDER BY created_at DESC')
        
        # Use fetchall() which works for both Turso and SQLite
        try:
            rows = result.fetchall()
        except Exception:
            rows = []
        
        contacts = []
        for row in rows:
            # Handle both Turso (tuple) and SQLite (Row object)
            if hasattr(row, '_asdict'):  # SQLite Row
                contact = row._asdict()
                # images excluded from list — fetch via /api/contacts/<id>/images
            elif isinstance(row, (list, tuple)):  # Turso tuple
                contact = {
                    'id': row[0],
                    'full_name': row[1],
                    'company': row[2],
                    'designation': row[3],
                    'phone': row[4],
                    'email': row[5],
                    'website': row[6],
                    'industry': row[7],
                    'sales_person': row[8] if len(row) > 8 else '',
                    'notes': row[9] if len(row) > 9 else '',
                    'created_at': row[10] if len(row) > 10 else row[9]
                }
            else:
                # Fallback: try to convert to dict
                try:
                    contact = dict(row)
                except:
                    contact = {'_raw': str(row)}
            contacts.append(contact)
        return jsonify(contacts), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/contacts/<int:contact_id>/images', methods=['GET'])
def get_contact_images(contact_id):
    """Get images for a specific contact (lazy load)"""
    db = get_db()
    try:
        result = db.execute('SELECT images FROM contacts WHERE id = ?', (contact_id,))
        try:
            rows = result.fetchall()
        except Exception:
            rows = []
        if not rows:
            return jsonify({'error': 'Contact not found'}), 404
        row = rows[0]
        images = json.loads(row[0]) if row[0] else []
        return jsonify({'images': images}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/contacts', methods=['POST'])
def create_contact():
    """Create a new contact"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate at least name or company exists
        if not data.get('fullName') and not data.get('company'):
            return jsonify({'error': 'Name or company is required'}), 400
        
        db = get_db()
        
        # Handle images - store as JSON (base64 in database for Vercel)
        images_json = json.dumps(data.get('images', []))
        
        result = db.execute('''
            INSERT INTO contacts (full_name, company, designation, phone, email, website, industry, sales_person, notes, images, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('fullName'),
            data.get('company'),
            data.get('designation'),
            data.get('phone'),
            data.get('email'),
            data.get('website'),
            data.get('industry'),
            data.get('salesPerson'),
            data.get('notes', ''),
            images_json,
            datetime.now().isoformat()
        ))
        
        # Get the last inserted row id - handle both SQLite and Turso
        try:
            contact_id = result.lastrowid if hasattr(result, 'lastrowid') else None
            if contact_id is None:
                # For Turso/libsql, try to get from SELECT
                try:
                    result = db.execute("SELECT last_insert_rowid() as id")
                    rows = list(result.rows) if hasattr(result, 'rows') else result.fetchall()
                    row = rows[0] if rows else None
                    contact_id = row[0] if row else None
                except Exception as e2:
                    contact_id = None
                    print(f"Warning: Could not get lastrowid: {e2}")
        except Exception as e:
            print(f"Warning: Could not get lastrowid: {e}")
            contact_id = None
        
        db.commit()
        db.close()
        
        if contact_id is None:
            return jsonify({'error': 'Failed to get contact ID after insert'}), 500
        
        return jsonify({'id': contact_id, 'message': 'Contact created successfully'}), 201
    except Exception as e:
        print(f"Error creating contact: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/contacts/<int:contact_id>', methods=['DELETE'])
def delete_contact(contact_id):
    """Delete a contact"""
    try:
        db = get_db()
        
        # Check if contact exists
        result = db.execute('SELECT id FROM contacts WHERE id = ?', (contact_id,))
        
        try:
            rows = result.fetchall()
        except:
            rows = list(result.rows) if hasattr(result, 'rows') else []
        
        if not rows:
            db.close()
            return jsonify({'error': 'Contact not found'}), 404
        
        db.execute('DELETE FROM contacts WHERE id = ?', (contact_id,))
        db.commit()
        db.close()
        
        return jsonify({'message': 'Contact deleted successfully'})
    except Exception as e:
        print(f"Error deleting contact: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/contacts/clear', methods=['DELETE'])
def clear_all_contacts():
    """Clear all contacts"""
    try:
        db = get_db()
        db.execute('DELETE FROM contacts')
        db.commit()
        db.close()
        
        return jsonify({'message': 'All contacts cleared'})
    except Exception as e:
        print(f"Error clearing contacts: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/contacts/<int:contact_id>', methods=['PATCH'])
def update_contact(contact_id):
    """Update a contact"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        db = get_db()
        
        # Build dynamic update query
        allowed_fields = ['full_name', 'company', 'designation', 'phone', 'email', 'website', 'industry', 'sales_person', 'notes', 'images']
        updates = []
        values = []
        
        field_mapping = {
            'fullName': 'full_name',
            'company': 'company',
            'designation': 'designation',
            'phone': 'phone',
            'email': 'email',
            'website': 'website',
            'industry': 'industry',
            'salesPerson': 'sales_person',
            'notes': 'notes',
            'images': 'images'
        }
        
        for key, db_field in field_mapping.items():
            if key in data:
                if db_field == 'images':
                    updates.append(f'{db_field} = ?')
                    values.append(json.dumps(data[key]))
                else:
                    updates.append(f'{db_field} = ?')
                    values.append(data[key])
        
        if not updates:
            db.close()
            return jsonify({'error': 'No valid fields to update'}), 400
        
        values.append(contact_id)
        query = f"UPDATE contacts SET {', '.join(updates)} WHERE id = ?"
        
        result = db.execute(query, values)
        
        try:
            rows_affected = result.rows_affected if hasattr(result, 'rows_affected') else result.rowcount
        except:
            rows_affected = 0
        
        db.commit()
        db.close()
        
        if rows_affected == 0:
            return jsonify({'error': 'Contact not found'}), 404
        
        return jsonify({'message': 'Contact updated successfully'}), 200
    except Exception as e:
        print(f"Error updating contact: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/contacts/bulk', methods=['PATCH'])
def bulk_update_contacts():
    """Bulk update contacts (delete, assign industry)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        action = data.get('action')
        contact_ids = data.get('ids', [])
        
        if not contact_ids:
            return jsonify({'error': 'No contact IDs provided'}), 400
        
        if not isinstance(contact_ids, list):
            return jsonify({'error': 'ids must be a list'}), 400
        
        db = get_db()
        
        if action == 'delete':
            placeholders = ','.join('?' * len(contact_ids))
            db.execute(f'DELETE FROM contacts WHERE id IN ({placeholders})', contact_ids)
            db.commit()
            db.close()
            return jsonify({'message': f'{len(contact_ids)} contacts deleted'}), 200
        
        elif action == 'update_industry':
            industry = data.get('industry', '')
            placeholders = ','.join('?' * len(contact_ids))
            db.execute(f'UPDATE contacts SET industry = ? WHERE id IN ({placeholders})', [industry] + contact_ids)
            db.commit()
            db.close()
            return jsonify({'message': f'{len(contact_ids)} contacts updated'}), 200
        
        elif action == 'delete_all':
            db.execute('DELETE FROM contacts')
            db.commit()
            db.close()
            return jsonify({'message': 'All contacts deleted'}), 200
        
        else:
            db.close()
            return jsonify({'error': f'Unknown action: {action}'}), 400
    
    except Exception as e:
        print(f"Error in bulk update: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    try:
        from datetime import datetime, timedelta
        
        db = get_db()
        
        # Total contacts
        result = db.execute('SELECT COUNT(*) as count FROM contacts')
        try:
            rows = result.fetchall()
            total = rows[0][0] if rows else 0
        except:
            total = 0
        
        # Contacts added today
        today = datetime.now().strftime('%Y-%m-%d')
        result = db.execute('SELECT COUNT(*) as count FROM contacts WHERE date(created_at) = ?', (today,))
        try:
            rows = result.fetchall()
            today_count = rows[0][0] if rows else 0
        except:
            today_count = 0
        
        # Contacts added this week
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        result = db.execute('SELECT COUNT(*) as count FROM contacts WHERE date(created_at) >= ?', (week_ago,))
        try:
            rows = result.fetchall()
            week_count = rows[0][0] if rows else 0
        except:
            week_count = 0
        
        # Contacts added this month
        month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        result = db.execute('SELECT COUNT(*) as count FROM contacts WHERE date(created_at) >= ?', (month_ago,))
        try:
            rows = result.fetchall()
            month_count = rows[0][0] if rows else 0
        except:
            month_count = 0
        
        # Contacts by industry
        result = db.execute('''
            SELECT industry, COUNT(*) as count 
            FROM contacts 
            WHERE industry IS NOT NULL AND industry != ''
            GROUP BY industry
            ORDER BY count DESC
        ''')
        try:
            rows = result.fetchall()
            by_industry = {row[0]: row[1] for row in rows}
        except:
            by_industry = {}
        
        # Recent contacts (last 5)
        result = db.execute('''
            SELECT id, full_name, company, industry, created_at 
            FROM contacts 
            ORDER BY created_at DESC 
            LIMIT 5
        ''')
        try:
            rows = result.fetchall()
            recent = []
            for row in rows:
                recent.append({
                    'id': row[0],
                    'full_name': row[1] or 'Unknown',
                    'company': row[2] or '',
                    'industry': row[3] or '',
                    'created_at': row[4]
                })
        except:
            recent = []
        
        db.close()
        
        return jsonify({
            'total': total,
            'today': today_count,
            'this_week': week_count,
            'this_month': month_count,
            'by_industry': by_industry,
            'recent': recent
        }), 200
    
    except Exception as e:
        print(f"Error getting stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Industries API
@app.route('/api/industries', methods=['GET'])
def get_industries():
    """Get all industries"""
    db = get_db()
    try:
        result = db.execute('SELECT id, name, is_default FROM industries ORDER BY is_default DESC, name ASC')
        
        # Use fetchall() which works for both Turso and SQLite
        try:
            rows = result.fetchall()
        except Exception:
            rows = []
        
        industries = []
        for row in rows:
            # Handle both Turso (tuple) and SQLite (Row object)
            if hasattr(row, '_asdict'):  # SQLite Row
                industries.append(row._asdict())
            elif isinstance(row, (list, tuple)):  # Turso tuple
                industries.append({
                    'id': row[0],
                    'name': row[1],
                    'is_default': row[2]
                })
            else:
                try:
                    industries.append(dict(row))
                except:
                    industries.append({'_raw': str(row)})
        return jsonify(industries), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/industries', methods=['POST'])
def create_industry():
    """Create a new industry"""
    data = request.get_json()
    
    if not data or not data.get('name'):
        return jsonify({'error': 'Industry name is required'}), 400
    
    industry_id = data.get('id', data['name'].lower().replace(' ', '-') + '_' + str(int(time.time())))
    
    db = get_db()
    
    try:
        db.execute(
            'INSERT INTO industries (id, name, is_default) VALUES (?, ?, 0)',
            (industry_id, data['name'])
        )
        db.commit()
        db.close()
        
        return jsonify({'id': industry_id, 'message': 'Industry created successfully'}), 201
    except Exception as e:
        db.close()
        error_msg = str(e)
        if 'UNIQUE constraint' in error_msg or 'duplicate key' in error_msg.lower():
            return jsonify({'error': 'Industry already exists'}), 400
        return jsonify({'error': str(e)}), 400

@app.route('/api/industries/<industry_id>', methods=['PUT'])
def update_industry(industry_id):
    """Update an industry"""
    try:
        data = request.get_json()
        
        if not data or not data.get('name'):
            return jsonify({'error': 'Industry name is required'}), 400
        
        db = get_db()
        
        result = db.execute('UPDATE industries SET name = ? WHERE id = ?', (data['name'], industry_id))
        
        # Check rows affected - Turso uses rows_affected, SQLite uses rowcount
        try:
            if hasattr(result, 'rows_affected'):
                rows_affected = result.rows_affected
            elif hasattr(result, 'rowcount'):
                rows_affected = result.rowcount
            else:
                rows_affected = 1  # Assume success if unknown
        except:
            rows_affected = 1  # Assume success if unknown
        
        if rows_affected == 0:
            db.close()
            return jsonify({'error': 'Industry not found'}), 404
        
        db.commit()
        db.close()
        
        return jsonify({'message': 'Industry updated successfully'})
    except Exception as e:
        print(f"Error updating industry: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/industries/<industry_id>', methods=['DELETE'])
def delete_industry(industry_id):
    """Delete an industry"""
    try:
        db = get_db()
        
        # Check if it's a default industry
        result = db.execute('SELECT is_default FROM industries WHERE id = ?', (industry_id,))
        
        try:
            rows = result.fetchall()
        except:
            rows = list(result.rows) if hasattr(result, 'rows') else []
        
        if not rows:
            db.close()
            return jsonify({'error': 'Industry not found'}), 404
        
        first_row = dict_from_row(rows[0])
        is_default = first_row.get('is_default', 0) if first_row else 0
        
        if is_default:
            db.close()
            return jsonify({'error': 'Cannot delete default industry'}), 400
        
        db.execute('DELETE FROM industries WHERE id = ?', (industry_id,))
        db.commit()
        db.close()
        
        return jsonify({'message': 'Industry deleted successfully'})
    except Exception as e:
        print(f"Error deleting industry: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Sales Persons API
@app.route('/api/sales_persons', methods=['GET'])
def get_sales_persons():
    """Get all sales persons"""
    db = get_db()
    try:
        result = db.execute('SELECT id, name, is_default FROM sales_persons ORDER BY is_default DESC, name ASC')
        
        # Use fetchall() which works for both Turso and SQLite
        try:
            rows = result.fetchall()
        except Exception:
            rows = []
        
        sales_persons = []
        for row in rows:
            # Handle both Turso (tuple) and SQLite (Row object)
            if hasattr(row, '_asdict'):  # SQLite Row
                sales_persons.append(row._asdict())
            elif isinstance(row, (list, tuple)):  # Turso tuple
                sales_persons.append({
                    'id': row[0],
                    'name': row[1],
                    'is_default': row[2]
                })
            else:
                try:
                    sales_persons.append(dict(row))
                except:
                    sales_persons.append({'_raw': str(row)})
        return jsonify(sales_persons), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/sales_persons', methods=['POST'])
def create_sales_person():
    """Create a new sales person"""
    data = request.get_json()
    
    if not data or not data.get('name'):
        return jsonify({'error': 'Sales person name is required'}), 400
    
    sales_person_id = data.get('id', 'sp_' + data['name'].lower().replace(' ', '_') + '_' + str(int(time.time())))
    
    db = get_db()
    
    try:
        db.execute(
            'INSERT INTO sales_persons (id, name, is_default) VALUES (?, ?, 0)',
            (sales_person_id, data['name'])
        )
        db.commit()
        db.close()
        
        return jsonify({'id': sales_person_id, 'message': 'Sales person created successfully'}), 201
    except Exception as e:
        db.close()
        error_msg = str(e)
        if 'UNIQUE constraint' in error_msg or 'duplicate key' in error_msg.lower():
            return jsonify({'error': 'Sales person already exists'}), 400
        return jsonify({'error': str(e)}), 400

@app.route('/api/sales_persons/<sales_person_id>', methods=['PUT'])
def update_sales_person(sales_person_id):
    """Update a sales person"""
    try:
        data = request.get_json()
        
        if not data or not data.get('name'):
            return jsonify({'error': 'Sales person name is required'}), 400
        
        db = get_db()
        
        result = db.execute('UPDATE sales_persons SET name = ? WHERE id = ?', (data['name'], sales_person_id))
        
        # Check rows affected - Turso uses rows_affected, SQLite uses rowcount
        try:
            if hasattr(result, 'rows_affected'):
                rows_affected = result.rows_affected
            elif hasattr(result, 'rowcount'):
                rows_affected = result.rowcount
            else:
                rows_affected = 1  # Assume success if unknown
        except:
            rows_affected = 1  # Assume success if unknown
        
        if rows_affected == 0:
            db.close()
            return jsonify({'error': 'Sales person not found'}), 404
        
        db.commit()
        db.close()
        
        return jsonify({'message': 'Sales person updated successfully'})
    except Exception as e:
        print(f"Error updating sales person: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sales_persons/<sales_person_id>', methods=['DELETE'])
def delete_sales_person(sales_person_id):
    """Delete a sales person"""
    try:
        db = get_db()
        
        # Check if it exists
        result = db.execute('SELECT is_default FROM sales_persons WHERE id = ?', (sales_person_id,))
        
        try:
            rows = result.fetchall()
        except:
            rows = list(result.rows) if hasattr(result, 'rows') else []
        
        if not rows:
            db.close()
            return jsonify({'error': 'Sales person not found'}), 404
        
        db.execute('DELETE FROM sales_persons WHERE id = ?', (sales_person_id,))
        db.commit()
        db.close()
        
        return jsonify({'message': 'Sales person deleted successfully'})
    except Exception as e:
        print(f"Error deleting sales person: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# OCR API - Synchronous for Vercel
@app.route('/api/ocr/batch', methods=['POST'])
def batch_ocr():
    """Submit multiple images and process synchronously (Vercel-friendly)"""
    data = request.get_json()
    
    if not data or not data.get('images'):
        return jsonify({'error': 'No images provided'}), 400
    
    images = data['images']
    results = []
    
    for image_data in images:
        try:
            # Convert base64 to proper format
            if ',' in image_data:
                base64_data = image_data.split(',')[1]
            else:
                base64_data = image_data
            
            # Process synchronously
            extracted_data = process_ocr_sync(base64_data)
            results.append(extracted_data)
                
        except Exception as e:
            results.append({'error': str(e)})
    
    # Merge results (combine data from multiple images)
    merged = {
        'fullName': '',
        'company': '',
        'designation': '',
        'phone': '',
        'email': '',
        'website': ''
    }
    
    for r in results:
        if isinstance(r, dict):
            for key in merged:
                if not merged[key] and r.get(key):
                    merged[key] = r[key]
    
    return jsonify(merged)

# File upload endpoint - base64 only (no filesystem writes in Vercel)
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads - returns base64 for storage in SQLite"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read and encode as base64 directly (no filesystem writes)
    image_base64 = base64.b64encode(file.read()).decode('utf-8')
    
    # Run OCR synchronously
    result = process_ocr_sync(image_base64)
    
    return jsonify({
        'filename': file.filename,
        'image_base64': image_base64,
        'extracted': result
    })

# Export API
@app.route('/api/export', methods=['GET'])
def export_excel():
    """Export contacts to Excel format (CSV for simplicity)"""
    try:
        import csv
        import io
        
        db = get_db()
        result = db.execute('SELECT * FROM contacts ORDER BY created_at DESC')
        
        try:
            rows = result.fetchall()
        except:
            rows = list(result.rows) if hasattr(result, 'rows') else []
        db.close()
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Name', 'Company', 'Designation', 'Phone', 'Email', 'Website', 'Industry', 'Notes', 'Created At'])
        
        # Data
        for row in rows:
            row_dict = dict_from_row(row)
            if row_dict is None:
                continue
            writer.writerow([
                row_dict.get('full_name', ''),
                row_dict.get('company', ''),
                row_dict.get('designation', ''),
                row_dict.get('phone', ''),
                row_dict.get('email', ''),
                row_dict.get('website', ''),
                row_dict.get('industry', ''),
                row_dict.get('notes', ''),
                row_dict.get('created_at', '')
            ])
        
        output.seek(0)
        
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=BSM_Contacts.csv'}
        )
    except Exception as e:
        print(f"Error exporting: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_type = "Turso" if TURSO_AVAILABLE else "SQLite"
    return jsonify({
        'status': 'ok',
        'database_type': db_type,
        'database_path': TURSO_DB_URL if TURSO_AVAILABLE else DATABASE_PATH,
        'vercel_mode': True
    })

# Initialize database on startup
init_database()

# Run locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

