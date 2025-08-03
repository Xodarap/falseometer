"""
Flask web application for Article Analyzer
"""

import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from datetime import datetime
from article_analyzer import ArticleAnalyzer

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

@app.route('/')
def index():
    """Main page with form to analyze articles."""
    # Get URL parameters to pre-fill the form
    url = request.args.get('url', '')
    max_sentences = request.args.get('max_sentences', '')
    max_claims = request.args.get('max_claims', '')
    skip_sentences = request.args.get('skip_sentences', '')
    
    return render_template('index.html', 
                         url=url,
                         max_sentences=max_sentences,
                         max_claims=max_claims,
                         skip_sentences=skip_sentences)

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Analyze an article and return results."""
    try:
        # Get form data
        input_method = request.form.get('input_method', 'url')
        url = request.form.get('url', '').strip()
        article_text = request.form.get('article_text', '').strip()
        max_sentences = request.form.get('max_sentences', '')
        max_claims = request.form.get('max_claims', '')
        skip_sentences = request.form.get('skip_sentences', '0')
        
        # Validate input
        if input_method == 'url':
            if not url:
                flash('Please provide a URL to analyze', 'error')
                return redirect(url_for('index'))
            article_source = url
        else:  # text method
            if not article_text:
                flash('Please provide article text to analyze', 'error')
                return redirect(url_for('index'))
            article_source = "Direct Text Input"
        
        # Convert parameters to integers with limits
        try:
            max_sentences = int(max_sentences) if max_sentences else 5
            max_claims = int(max_claims) if max_claims else 10
            skip_sentences = int(skip_sentences) if skip_sentences else 5
            
            # Enforce maximums
            if max_sentences > 50:
                flash('Maximum sentences limited to 50 for performance reasons', 'warning')
                max_sentences = 50
            if max_claims > 10:
                flash('Maximum claims per sentence limited to 10 for performance reasons', 'warning')
                max_claims = 10
                
        except ValueError:
            flash('Please provide valid numbers for sentence and claim limits', 'error')
            return redirect(url_for('index'))
        
        # Create analyzer and run analysis
        analyzer = ArticleAnalyzer()
        
        try:
            if input_method == 'url':
                results = analyzer.analyze_article(
                    url, 
                    max_sentences=max_sentences, 
                    max_claims=max_claims, 
                    skip_sentences=skip_sentences
                )
                source_url = url
            else:  # text method
                results = analyzer.analyze_text(
                    article_text,
                    max_sentences=max_sentences, 
                    max_claims=max_claims, 
                    skip_sentences=skip_sentences
                )
                source_url = "Direct Text Input"
        except Exception as e:
            flash(f'Analysis failed: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Prepare data for template
        sentences_data = []
        for analysis in results:
            sentence_data = {
                "sentence": analysis.sentence,
                "sentence_microlies": analysis.sentence_microlies,
                "claims": [
                    {
                        "text": claim.text,
                        "probability_interpreted": claim.probability_interpreted,
                        "probability_true": claim.probability_true,
                        "interpretation_explanation": claim.interpretation_explanation,
                        "truth_explanation": claim.truth_explanation,
                        "microlies": claim.microlies
                    }
                    for claim in analysis.claims
                ]
            }
            sentences_data.append(sentence_data)
        
        # Calculate summary statistics
        total_sentences = len(results)
        total_claims = sum(len(analysis.claims) for analysis in results)
        article_microlies = sum(analysis.sentence_microlies for analysis in results)
        
        summary = {
            "url": source_url,
            "total_sentences": total_sentences,
            "total_claims": total_claims,
            "article_microlies": article_microlies,
            "avg_claims_per_sentence": total_claims / total_sentences if total_sentences > 0 else 0,
            "avg_microlies_per_sentence": article_microlies / total_sentences if total_sentences > 0 else 0,
            "avg_microlies_per_claim": article_microlies / total_claims if total_claims > 0 else 0,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Include parameters for sharing
            "max_sentences": max_sentences,
            "max_claims": max_claims,
            "skip_sentences": skip_sentences
        }
        
        return render_template('results.html', 
                             sentences=sentences_data, 
                             summary=summary)
    
    except Exception as e:
        flash(f'Error analyzing article: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access."""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        max_sentences = data.get('max_sentences', 5)
        max_claims = data.get('max_claims', 10)
        skip_sentences = data.get('skip_sentences', 5)
        
        # Enforce maximums
        if max_sentences and max_sentences > 50:
            max_sentences = 50
        if max_claims and max_claims > 10:
            max_claims = 10
        
        # Create analyzer and run analysis
        analyzer = ArticleAnalyzer()
        results = analyzer.analyze_article(
            url, 
            max_sentences=max_sentences, 
            max_claims=max_claims, 
            skip_sentences=skip_sentences
        )
        
        # Format response
        sentences_data = []
        for analysis in results:
            sentence_data = {
                "sentence": analysis.sentence,
                "sentence_microlies": analysis.sentence_microlies,
                "claims": [
                    {
                        "text": claim.text,
                        "probability_interpreted": claim.probability_interpreted,
                        "probability_true": claim.probability_true,
                        "interpretation_explanation": claim.interpretation_explanation,
                        "truth_explanation": claim.truth_explanation,
                        "microlies": claim.microlies
                    }
                    for claim in analysis.claims
                ]
            }
            sentences_data.append(sentence_data)
        
        # Calculate summary
        total_sentences = len(results)
        total_claims = sum(len(analysis.claims) for analysis in results)
        article_microlies = sum(analysis.sentence_microlies for analysis in results)
        
        response = {
            "article_microlies": article_microlies,
            "total_sentences": total_sentences,
            "total_claims": total_claims,
            "sentences": sentences_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze')
def analyze_get():
    """Analyze an article via GET request with URL parameters."""
    try:
        # Get URL parameters
        url = request.args.get('url', '').strip()
        max_sentences = request.args.get('max_sentences', '5')
        max_claims = request.args.get('max_claims', '10')
        skip_sentences = request.args.get('skip_sentences', '3')
        
        # Validate URL
        if not url:
            flash('Please provide a URL to analyze', 'error')
            return redirect(url_for('index'))
        
        # Convert parameters to integers with limits
        try:
            max_sentences = int(max_sentences) if max_sentences else 5
            max_claims = int(max_claims) if max_claims else 10
            skip_sentences = int(skip_sentences) if skip_sentences else 3
            
            # Enforce maximums
            if max_sentences > 50:
                max_sentences = 50
            if max_claims > 10:
                max_claims = 10
                
        except ValueError:
            flash('Please provide valid numbers for sentence and claim limits', 'error')
            return redirect(url_for('index'))
        
        # Create analyzer and run analysis
        analyzer = ArticleAnalyzer()
        
        try:
            results = analyzer.analyze_article(
                url, 
                max_sentences=max_sentences, 
                max_claims=max_claims, 
                skip_sentences=skip_sentences
            )
        except Exception as e:
            flash(f'Analysis failed: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Prepare data for template (same as POST route)
        sentences_data = []
        for analysis in results:
            sentence_data = {
                "sentence": analysis.sentence,
                "sentence_microlies": analysis.sentence_microlies,
                "claims": [
                    {
                        "text": claim.text,
                        "probability_interpreted": claim.probability_interpreted,
                        "probability_true": claim.probability_true,
                        "interpretation_explanation": claim.interpretation_explanation,
                        "truth_explanation": claim.truth_explanation,
                        "microlies": claim.microlies
                    }
                    for claim in analysis.claims
                ]
            }
            sentences_data.append(sentence_data)
        
        # Calculate summary statistics
        total_sentences = len(results)
        total_claims = sum(len(analysis.claims) for analysis in results)
        article_microlies = sum(analysis.sentence_microlies for analysis in results)
        
        summary = {
            "url": url,
            "total_sentences": total_sentences,
            "total_claims": total_claims,
            "article_microlies": article_microlies,
            "avg_claims_per_sentence": total_claims / total_sentences if total_sentences > 0 else 0,
            "avg_microlies_per_sentence": article_microlies / total_sentences if total_sentences > 0 else 0,
            "avg_microlies_per_claim": article_microlies / total_claims if total_claims > 0 else 0,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Include parameters for sharing
            "max_sentences": max_sentences,
            "max_claims": max_claims,
            "skip_sentences": skip_sentences
        }
        
        return render_template('results.html', 
                             sentences=sentences_data, 
                             summary=summary)
    
    except Exception as e:
        flash(f'Error analyzing article: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)