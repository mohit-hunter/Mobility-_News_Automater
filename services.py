import feedparser
import pandas as pd
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import requests
from bs4 import BeautifulSoup
import time
import re
import logging
import json
import os

class ScraperService:
    def __init__(self, master_json_path='master_news_archive.json', feeds_json_path='feeds.json'):
        self.master_json_path = master_json_path
        self.feeds_json_path = feeds_json_path
        self.similarity_threshold = 85

    def get_feeds(self):
        """Load feeds from JSON file."""
        if not os.path.exists(self.feeds_json_path):
            return []
        try:
            with open(self.feeds_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading feeds: {e}")
            return []

    def save_feeds(self, feeds):
        """Save feeds to JSON file."""
        with open(self.feeds_json_path, 'w', encoding='utf-8') as f:
            json.dump(feeds, f, indent=2)

    def add_feed(self, name, url):
        """Add a new feed."""
        feeds = self.get_feeds()
        # Check if exists
        for feed in feeds:
            if feed['url'] == url:
                return False, "Feed URL already exists"
            if feed['name'] == name:
                return False, "Feed name already exists"
        
        feeds.append({"name": name, "url": url})
        self.save_feeds(feeds)
        return True, "Feed added successfully"

    def remove_feed(self, name):
        """Remove a feed by name."""
        feeds = self.get_feeds()
        initial_len = len(feeds)
        feeds = [f for f in feeds if f['name'] != name]
        if len(feeds) < initial_len:
            self.save_feeds(feeds)
            return True, "Feed removed"
        return False, "Feed not found"

    def load_master_json(self):
        """Load existing master JSON archive or return empty list."""
        if os.path.exists(self.master_json_path):
            try:
                with open(self.master_json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Error loading master JSON: {e}")
                return []
        return []

    def get_news(self, start_date=None, end_date=None, search_query=None):
        """Retrieve news from master archive with filtering."""
        data = self.load_master_json()
        
        # Convert to dataframe for easier filtering if needed, or just iterate
        # Assuming date format YYYY-MM-DD
        
        filtered_data = []
        for item in data:
            item_date = item.get('Date')
            if not item_date:
                continue
            
            # Simple string comparison works for YYYY-MM-DD
            if start_date and item_date < start_date:
                continue
            if end_date and item_date > end_date:
                continue
                
            if search_query:
                query = search_query.lower()
                if (query not in item.get('Executive Headline', '').lower() and 
                    query not in item.get('Summary', '').lower() and
                    query not in item.get('Source', '').lower()):
                    continue
            
            filtered_data.append(item)
            
        # Sort by date descending
        filtered_data.sort(key=lambda x: x['Date'], reverse=True)
        return filtered_data

    def run_scraper(self, days_back=7):
        """Run the scraping process."""
        feeds = self.get_feeds()
        # Convert list of dicts to dict for compatibility with original logic if needed, 
        # but better to adapt logic to list of dicts.
        
        date_to = datetime.now()
        date_from = date_to - timedelta(days=days_back)
        
        logging.info(f"Collecting news from {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}")
        
        news_data = []
        
        for feed_item in feeds:
            source_name = feed_item['name']
            url = feed_item['url']
            
            logging.info(f"Fetching from {source_name}...")
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # Parse publication date
                    pub_datetime = None
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_datetime = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_datetime = datetime(*entry.updated_parsed[:6])
                        else:
                            continue
                    except Exception as e:
                        logging.warning(f"  Skipping entry due to date parse error: {e}")
                        continue
                    
                    # Filter by date range
                    if not (date_from <= pub_datetime <= date_to):
                        continue
                    
                    # Extract content
                    title = entry.get('title', '').strip()
                    summary = entry.get('summary', '') or entry.get('description', '')
                    link = entry.get('link', '')
                    
                    # Clean HTML tags from summary
                    if summary:
                        summary = BeautifulSoup(summary, 'html.parser').get_text()
                        summary = re.sub(r'\s+', ' ', summary).strip()
                    
                    news_data.append({
                        'source': source_name,
                        'title': title,
                        'summary': summary[:500],  # Limit summary length
                        'link': link,
                        'pub_date': pub_datetime,
                    })
                
                logging.info(f"  Found {len([d for d in news_data if d['source'] == source_name])} articles")
                time.sleep(1)  # Be polite
                
            except Exception as e:
                logging.error(f"  Error fetching {source_name}: {e}")

        if not news_data:
            return {"status": "success", "new_articles": 0, "total_articles": 0}

        # Process Data
        df = pd.DataFrame(news_data)
        df = df.sort_values('pub_date', ascending=False).reset_index(drop=True)
        
        # Deduplication
        logging.info("Deduplicating...")
        df['cluster'] = -1
        cluster_id = 0
        
        for i in range(len(df)):
            if df.loc[i, 'cluster'] != -1:
                continue
            
            df.loc[i, 'cluster'] = cluster_id
            title_i = df.loc[i, 'title']
            
            for j in range(i + 1, len(df)):
                if df.loc[j, 'cluster'] != -1:
                    continue
                
                title_j = df.loc[j, 'title']
                similarity = fuzz.token_sort_ratio(title_i, title_j)
                
                if similarity > self.similarity_threshold:
                    df.loc[j, 'cluster'] = cluster_id
            
            cluster_id += 1

        # Create Executive Summary Columns
        df['executive_headline'] = df.apply(lambda row: self._create_executive_headline(row['title'], row['summary']), axis=1)
        df['executive_summary'] = df.apply(lambda row: self._create_executive_summary(row['summary'], row['title']), axis=1)
        
        # Prepare Output
        output_df = df[[
            'pub_date', 'executive_headline', 'executive_summary', 'link', 
            'source', 'cluster'
        ]].copy()
        
        output_df['pub_date'] = output_df['pub_date'].dt.strftime('%Y-%m-%d')
        output_df = output_df.rename(columns={
            'pub_date': 'Date',
            'executive_headline': 'Executive Headline',
            'executive_summary': 'Summary',
            'link': 'Source URL',
            'source': 'Source',
            'cluster': 'Cluster ID'
        })
        
        # Save results
        return self._save_results(output_df, date_from, date_to)

    def _create_executive_headline(self, title, summary):
        """Rephrase headline to executive-friendly format"""
        title = re.sub(r'\[.*?\]', '', title).strip()
        title = re.sub(r'\s+', ' ', title)
        if len(title) > 100:
            title = title[:97] + "..."
        return title

    def _create_executive_summary(self, summary, title):
        """Create concise 3-5 sentence strategic summary"""
        if not summary or len(summary) < 50:
            return f"Strategic update: {title}"
        
        summary = re.sub(r'\s+', ' ', summary).strip()
        sentences = re.split(r'[.!?]+', summary)
        executive_summary = '. '.join([s.strip() for s in sentences[:5] if s.strip()])
        
        if len(executive_summary) > 500:
            executive_summary = executive_summary[:497] + "..."
        
        return executive_summary + "." if executive_summary and not executive_summary.endswith('.') else executive_summary

    def _save_results(self, output_df, date_from, date_to):
        execution_timestamp = datetime.now().isoformat()
        master_data = self.load_master_json()
        
        existing_links = set(article.get('Source URL') for article in master_data if 'Source URL' in article)
        
        new_records = []
        for _, row in output_df.iterrows():
            if row['Source URL'] not in existing_links:
                record = {
                    'Date': row['Date'],
                    'Executive Headline': row['Executive Headline'],
                    'Summary': row['Summary'],
                    'Source URL': row['Source URL'],
                    'Source': row['Source'],
                    'Cluster ID': int(row['Cluster ID']),
                    'execution_timestamp': execution_timestamp
                }
                new_records.append(record)
        
        if new_records:
            master_data.extend(new_records)
            self._save_json_file(master_data, self.master_json_path)
            
            # Also save current run file
            current_filename = f"mobility_news_current_{date_from.strftime('%Y%m%d')}_{date_to.strftime('%Y%m%d')}.json"
            self._save_json_file(new_records, current_filename)
            
        return {
            "status": "success",
            "new_articles": len(new_records),
            "total_articles": len(master_data),
            "current_run_file": f"mobility_news_current_{date_from.strftime('%Y%m%d')}_{date_to.strftime('%Y%m%d')}.json" if new_records else None
        }

    def _save_json_file(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
