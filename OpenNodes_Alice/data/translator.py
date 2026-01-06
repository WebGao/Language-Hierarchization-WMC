import json
import os
import re
from typing import List, Dict, Optional
import requests
import time

class DeepSeekTranslator:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """
        Initialize the DeepSeek translator.
        
        Args:
            api_key: DeepSeek API key.
            base_url: API base URL, defaults to official DeepSeek API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def read_file(self, file_path: str) -> str:
        """
        Read a local text file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            The content of the file as a string.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
            raise
        except Exception as e:
            print(f"Error occurred while reading the file: {e}")
            raise
    
    def split_into_chapters(self, text: str) -> List[Dict[str, str]]:
        """
        Divide the text into chapters based on formats like CHAPTER I., CHAPTER II., etc.
        
        Args:
            text: Full text content.
            
        Returns:
            A list of chapters, each containing a title and content.
        """
        chapters = []
        
        # Optimize chapter splitting pattern: match CHAPTER I., CHAPTER II., etc.
        # Supports Roman numerals I, II, III, IV, V, VI, VII, VIII, IX, X, etc.
        chapter_pattern = r'(CHAPTER\s+[IVXLCDM]+\.)'
        
        # Find positions of all chapter titles
        matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
        
        if not matches:
            print("No CHAPTER X. format detected, trying alternative formats...")
            # Try other possible formats
            alternative_patterns = [
                r'(CHAPTER\s+\d+\.)',  # CHAPTER 1.
                r'(Chapter\s+[IVXLCDM]+\.)',  # Chapter I.
                r'(Chapter\s+\d+\.)',  # Chapter 1.
            ]
            
            for pattern in alternative_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    print(f"Detected format: {pattern}")
                    break
        
        if not matches:
            print("No standard chapter format detected, treating the entire text as one chapter.")
            return [{
                'title': '完整文本',
                'content': text.strip(),
                'chapter_number': 1
            }]
        
        print(f"Detected {len(matches)} chapters")
        
        # Split text based on chapter titles
        for i, match in enumerate(matches):
            chapter_title = match.group().strip()
            
            # Get start position
            start_pos = match.start()
            
            # Determine content range
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            chapter_content = text[start_pos:end_pos].strip()
            
            # Extract Roman numerals
            roman_num = re.search(r'[IVXLCDM]+', chapter_title, re.IGNORECASE)
            if roman_num:
                chapter_number = self._roman_to_int(roman_num.group())
            else:
                chapter_number = i + 1
            
            chapters.append({
                'title': chapter_title,
                'content': chapter_content,
                'chapter_number': chapter_number,
                'roman_num': roman_num.group() if roman_num else str(i + 1)
            })
            
            print(f"  Chapter {i + 1}: {chapter_title} (Roman Numeral: {roman_num.group() if roman_num else 'N/A'})")
        
        return chapters
    
    def _roman_to_int(self, roman: str) -> int:
        """
        Convert Roman numerals to integers.
        
        Args:
            roman: Roman numeral string.
            
        Returns:
            The corresponding integer value.
        """
        roman_dict = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        roman = roman.upper()
        total = 0
        prev_value = 0
        
        for char in reversed(roman):
            value = roman_dict.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
            
        return total
    
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Call DeepSeek API to translate text.
        
        Args:
            text: Text to be translated.
            target_language: Target language (e.g., Chinese, French, Spanish, etc.).
            
        Returns:
            The translated text.
        """
        # Construct API request
        url = f"{self.base_url}/chat/completions"
        
        prompt = f"""请将以下英文文本翻译成{target_language}。

翻译要求：
1. 保持原文的格式、段落、章节标题和标点符号
2. 人名、地名等专有名词保持原样，首次出现时可在括号内注明音译
3. 确保翻译准确自然，符合{target_language}的表达习惯
4. 不要添加任何额外的解释、说明 or 注释
5. 保持CHAPTER标题格式不变，只翻译内容

英文文本：
{text}

{target_language}翻译："""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # Lower temperature for more stable translations
            "max_tokens": 4000,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            
            # Ensure translation starts with the CHAPTER title
            if not translated_text.startswith("CHAPTER"):
                # Search for CHAPTER title
                chapter_match = re.search(r'(CHAPTER\s+[IVXLCDM]+\.)', text)
                if chapter_match:
                    chapter_title = chapter_match.group()
                    if chapter_title not in translated_text:
                        translated_text = f"{chapter_title}\n\n{translated_text}"
            
            return translated_text
            
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status code: {e.response.status_code}")
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response content: {e.response.text[:200]}")
            raise
        except KeyError as e:
            print(f"Error parsing API response: {e}")
            print(f"API Response: {result}")
            raise
        except Exception as e:
            print(f"Error occurred during translation: {e}")
            raise
    
    def translate_chapter(self, chapter: Dict, target_language: str, 
                         max_chunk_length: int = 2500) -> Dict:
        """
        Translate a single chapter, handling long text chunking.
        
        Args:
            chapter: Chapter dictionary.
            target_language: Target language.
            max_chunk_length: Maximum length for each translation chunk.
            
        Returns:
            Chapter dictionary containing translated content.
        """
        print(f"\nTranslating: {chapter['title']} ({len(chapter['content'])} characters)")
        
        content = chapter['content']
        
        # If content is too long, translate in chunks
        if len(content) > max_chunk_length:
            print(f"  Chapter content is long, translating in chunks...")
            chunks = self._split_into_chunks(content, max_chunk_length)
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"  Translating chunk {i + 1}/{len(chunks)}...")
                try:
                    # Add delay to avoid API limits
                    if i > 0:
                        time.sleep(1)
                    
                    translated_chunk = self.translate_text(chunk, target_language)
                    translated_chunks.append(translated_chunk)
                except Exception as e:
                    print(f"  Chunk {i + 1} failed: {e}")
                    # If translation fails, keep original and mark it
                    translated_chunks.append(f"[Translation failed, original kept]\n{chunk}")
                
            translated_content = '\n\n'.join(translated_chunks)
        else:
            translated_content = self.translate_text(content, target_language)
        
        # Extract the translated title
        translated_title = self._extract_translated_title(translated_content, chapter['title'])
        
        return {
            'original_title': chapter['title'],
            'translated_title': translated_title,
            'original_content': content,
            'translated_content': translated_content,
            'chapter_number': chapter['chapter_number'],
            'roman_num': chapter.get('roman_num', str(chapter['chapter_number']))
        }
    
    def _extract_translated_title(self, translated_content: str, original_title: str) -> str:
        """
        Extract the chapter title from the translated content.
        
        Args:
            translated_content: Translated content.
            original_title: Original title.
            
        Returns:
            Extracted title.
        """
        # Try to extract CHAPTER title from translated content
        title_patterns = [
            r'(CHAPTER\s+[IVXLCDM]+\.)',
            r'(第\s*[一二三四五六七八九十]+\s*章)',
            r'(Chapter\s+[IVXLCDM]+\.)',
            r'(章节\s+[IVXLCDM]+\.)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, translated_content, re.IGNORECASE)
            if match:
                return match.group()
        
        # If not found, return original title
        return original_title
    
    def _split_into_chunks(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks suitable for translation, keeping paragraphs intact.
        
        Args:
            text: Original text.
            max_length: Maximum length for each chunk.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        
        # Split by paragraphs
        paragraphs = re.split(r'(\n\s*\n)', text)
        
        current_chunk = ""
        for i in range(0, len(paragraphs), 2):
            paragraph = paragraphs[i]
            separator = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
            
            # If current chunk plus paragraph doesn't exceed limit
            if len(current_chunk) + len(paragraph) + len(separator) <= max_length:
                current_chunk += paragraph + separator
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If a single paragraph exceeds max length, split by sentences
                if len(paragraph) > max_length:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 2 <= max_length:
                            if temp_chunk:
                                temp_chunk += ' ' + sentence
                            else:
                                temp_chunk = sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    current_chunk = separator
                else:
                    current_chunk = paragraph + separator
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def save_translation(self, translated_chapters: List[Dict], 
                        output_dir: str, target_language: str, 
                        save_format: str = 'both'):
        """
        Save translation results.
        
        Args:
            translated_chapters: List of translated chapters.
            output_dir: Output directory.
            target_language: Target language.
            save_format: Save format ('txt', 'json', or 'both').
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        safe_lang = re.sub(r'[^\w\s-]', '', target_language).replace(' ', '_').lower()
        base_filename = f"alice_{safe_lang}"
        
        # Save complete translation file
        if save_format in ['txt', 'both']:
            txt_path = os.path.join(output_dir, f"{base_filename}_complete.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                for chapter in translated_chapters:
                    f.write(f"{chapter['translated_title']}\n")
                    f.write("-" * 60 + "\n\n")
                    f.write(chapter['translated_content'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            print(f"\nComplete translation saved as TXT: {txt_path}")
        
        # Save JSON format (contains source and translation)
        if save_format in ['json', 'both']:
            json_path = os.path.join(output_dir, f"{base_filename}_bilingual.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(translated_chapters, f, ensure_ascii=False, indent=2)
            
            print(f"Bilingual version saved as JSON: {json_path}")
        
        # Save clean translated file (no extra separators)
        txt_clean_path = os.path.join(output_dir, f"{base_filename}_clean.txt")
        with open(txt_clean_path, 'w', encoding='utf-8') as f:
            for chapter in translated_chapters:
                f.write(chapter['translated_content'])
                f.write("\n\n")
        
        print(f"Clean translation saved as: {txt_clean_path}")
        
        # Save individual chapter files
        chapters_dir = os.path.join(output_dir, "individual_chapters")
        os.makedirs(chapters_dir, exist_ok=True)
        
        print(f"\nSaving individual chapter files...")
        for chapter in translated_chapters:
            chapter_num = chapter['chapter_number']
            roman_num = chapter.get('roman_num', str(chapter_num))
            
            # Save translation
            translated_filename = f"chapter_{roman_num}_{safe_lang}.txt"
            translated_path = os.path.join(chapters_dir, translated_filename)
            
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(chapter['translated_content'])
            
            # Save bilingual version
            bilingual_filename = f"chapter_{roman_num}_bilingual.txt"
            bilingual_path = os.path.join(chapters_dir, bilingual_filename)
            
            with open(bilingual_path, 'w', encoding='utf-8') as f:
                f.write("=" * 40 + " Original English " + "=" * 40 + "\n")
                f.write(chapter['original_content'])
                f.write("\n\n" + "=" * 40 + f" {target_language} Translation " + "=" * 40 + "\n")
                f.write(chapter['translated_content'])
        
        print(f"Individual chapter files saved to: {chapters_dir}")
        
        # Generate summary report
        self._generate_summary(translated_chapters, output_dir, target_language)
    
    def _generate_summary(self, chapters: List[Dict], output_dir: str, target_language: str):
        """
        Generate translation summary report.
        
        Args:
            chapters: List of chapters.
            output_dir: Output directory.
            target_language: Target language.
        """
        report_path = os.path.join(output_dir, "translation_summary.txt")
        
        total_original_chars = sum(len(chapter['original_content']) for chapter in chapters)
        total_translated_chars = sum(len(chapter['translated_content']) for chapter in chapters)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Alice in Wonderland {target_language} Translation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Translation Info:\n")
            f.write(f"- Target Language: {target_language}\n")
            f.write(f"- Chapter Count: {len(chapters)}\n")
            f.write(f"- Original Character Count: {total_original_chars:,}\n")
            f.write(f"- Translated Character Count: {total_translated_chars:,}\n")
            f.write(f"- Translation Ratio: {total_translated_chars/total_original_chars:.2f}\n\n")
            
            f.write("Chapter Details:\n")
            f.write("-" * 60 + "\n")
            
            for chapter in chapters:
                orig_len = len(chapter['original_content'])
                trans_len = len(chapter['translated_content'])
                ratio = trans_len / orig_len if orig_len > 0 else 0
                
                f.write(f"Chapter {chapter['chapter_number']} ({chapter.get('roman_num', 'N/A')})\n")
                f.write(f"  Title: {chapter['original_title']} → {chapter['translated_title']}\n")
                f.write(f"  Source Length: {orig_len:,} chars\n")
                f.write(f"  Translation Length: {trans_len:,} chars\n")
                f.write(f"  Length Ratio: {ratio:.2f}\n")
                f.write("-" * 40 + "\n")
            
            f.write(f"\nReport Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Translation summary report generated: {report_path}")

def main():
    """Main function."""
    # Configuration parameters
    API_KEY = ""  # Replace with your API key
    INPUT_FILE = "alice_english.txt"  # Input file path
    lanList = ['Chinese', 'French', 'German', 'Russian', 'Japanese', 'Spanish', 'Italian']
    TARGET_LANGUAGE = lanList[6] # Target language index: 0, 1, 2, 3, 4, 5, 6
    OUTPUT_DIR = "alice_translation"  # Output directory
    
    print("Alice in Wonderland Chapter Translation Tool")
    print("=" * 50)
    
    # Initialize translator
    translator = DeepSeekTranslator(api_key=API_KEY)
    
    try:
        # 1. Read file
        print("Step 1: Reading file...")
        text_content = translator.read_file(INPUT_FILE)
        print(f"✓ File reading completed, total characters: {len(text_content):,}")
        
        # 2. Split chapters
        print("\nStep 2: Analyzing chapter structure...")
        chapters = translator.split_into_chapters(text_content)
        print(f"✓ Total of {len(chapters)} chapters identified.")
        
        # 3. Translate each chapter
        print(f"\nStep 3: Starting translation to {TARGET_LANGUAGE}...")
        print("=" * 50)
        
        translated_chapters = []
        
        for i, chapter in enumerate(chapters):
            try:
                print(f"\n[{i + 1}/{len(chapters)}] ", end="")
                translated_chapter = translator.translate_chapter(
                    chapter, 
                    TARGET_LANGUAGE
                )
                translated_chapters.append(translated_chapter)
                print(f"✓ Done: {chapter['title']}")
                
                # Add delay between chapters
                if i < len(chapters) - 1:
                    time.sleep(2)  # Avoid API rate limiting
                    
            except Exception as e:
                print(f"\n✗ Chapter translation failed: {chapter['title']}")
                print(f"   Error: {e}")
                print("   Skipping this chapter and continuing...")
                
                # Add placeholder for failed chapter
                translated_chapters.append({
                    'original_title': chapter['title'],
                    'translated_title': chapter['title'] + " [Translation Failed]",
                    'original_content': chapter['content'],
                    'translated_content': f"【Translation Failed】\n\n{chapter['content']}",
                    'chapter_number': chapter['chapter_number'],
                    'roman_num': chapter.get('roman_num', str(chapter['chapter_number']))
                })
        
        # 4. Save results
        print("\n" + "=" * 50)
        print("Step 4: Saving translation results...")
        translator.save_translation(
            translated_chapters,
            OUTPUT_DIR,
            TARGET_LANGUAGE,
            save_format='both'
        )
        
        print("\n" + "=" * 50)
        print("🎉 Translation task completed!")
        print(f"All files saved to: {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

def quick_translate():
    """Quick translate function (simplified version)."""
    API_KEY = input("Enter DeepSeek API Key: ").strip()
    TARGET_LANGUAGE = input("Enter target language (e.g., Chinese, Japanese, French): ").strip()
    INPUT_FILE = "alice_english.txt"
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} does not exist!")
        return
    
    translator = DeepSeekTranslator(api_key=API_KEY)
    
    # Read and split chapters
    text = translator.read_file(INPUT_FILE)
    chapters = translator.split_into_chapters(text)
    
    print(f"\nStarting translation for {len(chapters)} chapters...")
    
    # Translate only first 3 chapters as sample
    sample_chapters = chapters[:3]
    translated = []
    
    for chapter in sample_chapters:
        print(f"Translating: {chapter['title']}")
        try:
            result = translator.translate_chapter(chapter, TARGET_LANGUAGE)
            translated.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Save sample
    output_dir = "sample_translation"
    translator.save_translation(translated, output_dir, TARGET_LANGUAGE, save_format='txt')
    
    print(f"\nSample translation saved to: {output_dir}")

if __name__ == "__main__":
    print("Please select mode:")
    print("1. Full Translation (All chapters)")
    print("2. Quick Test (First 3 chapters only)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_translate()
    else:
        main()