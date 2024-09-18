import os
import sys
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from collections import Counter

try:
    import pandas as pd
except ImportError:
    print("pandasライブラリがインストールされていません。以下のコマンドでインストールしてください。")
    print("pip install pandas==1.5.0")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("PyYAMLライブラリがインストールされていません。以下のコマンドでインストールしてください。")
    print("pip install PyYAML==6.0")
    sys.exit(1)

try:
    from janome.tokenizer import Tokenizer
except ImportError:
    print("janomeライブラリがインストールされていません。以下のコマンドでインストールしてください。")
    print("pip install janome==0.4.2")
    sys.exit(1)

def load_config(config_path='config.yaml'):
    """
    設定ファイルを読み込む
    """
    if not os.path.exists(config_path):
        print(f"エラー: 設定ファイル '{config_path}' が見つかりません。")
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

def copy_history_file():
    """
    Chromeの履歴ファイルを一時的な場所にコピーする。
    """
    user_path = os.path.expanduser('~')
    history_path = os.path.join(user_path, 'AppData', 'Local', 'Google', 'Chrome', 'User Data', 'Default', 'History')
    temp_history_path = 'History_copy'
    shutil.copy2(history_path, temp_history_path)
    return temp_history_path

def get_target_date(date_str='today', timezone_info=None):
    """
    対象の日付を取得する。
    """
    if timezone_info is None:
        timezone_info = datetime.now().astimezone().tzinfo
    if date_str == 'today':
        return datetime.now(timezone_info)
    elif date_str == 'yesterday':
        return (datetime.now(timezone_info) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        return datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone_info)


def calculate_time_stamps(target_date, config):
    """
    対象の日付に基づいてタイムスタンプを計算する。
    """
    # Chromeのエポックタイムの開始日（UTC）
    epoch_start = datetime(1601, 1, 1, tzinfo=timezone.utc)
    
    # ローカルタイムゾーンを取得
    local_timezone = datetime.now().astimezone().tzinfo
    
    # target_dateの0時（ローカルタイムゾーン）を取得
    target_local = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=local_timezone)
    
    # target_localをUTCに変換
    target_utc = target_local.astimezone(timezone.utc)
    
    # 設定ファイルから日付のフォーマットを取得
    date_format = config.get('date_format', '%Y-%m-%d')  # デフォルトは '%Y-%m-%d'
    
    # 日付を指定された形式でフォーマット
    date_str = target_local.strftime(date_format)
    
    # タイムスタンプを計算
    time_delta = target_utc - epoch_start
    micros = int(time_delta.total_seconds() * 1e6)
    
    # 次の日のタイムスタンプを計算（範囲指定のため）
    next_day_local = target_local + timedelta(days=1)
    next_day_utc = next_day_local.astimezone(timezone.utc)
    time_delta_next = next_day_utc - epoch_start
    micros_next = int(time_delta_next.total_seconds() * 1e6)
    
    return micros, micros_next, date_str

def fetch_history_data(temp_history_path, micros_start, micros_end):
    """
    履歴データベースから指定の期間のデータを取得する。
    """
    try:
        conn = sqlite3.connect(temp_history_path)
    except sqlite3.Error as e:
        print(f"データベースに接続できませんでした: {e}")
        sys.exit(1)
    try:
        query = f'''
        SELECT
            urls.url,
            urls.title,
            visits.visit_time,
            visits.from_visit
        FROM
            urls, visits
        WHERE
            urls.id = visits.url
            AND visits.visit_time >= {micros_start}
            AND visits.visit_time < {micros_end}
        ORDER BY
            visits.visit_time ASC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"データの取得中にエラーが発生しました: {e}")
        conn.close()
        sys.exit(1)

def process_history_data(df, target_date, config):
    """
    取得した履歴データを加工・分析する。
    """
    if df.empty:
        print("データが取得できませんでした。")
        return None, None, None, None, None, None
    else:
        # タイムゾーンの設定
        local_timezone = target_date.tzinfo or datetime.now().astimezone().tzinfo

        # タイムスタンプの変換
        # ChromeのタイムスタンプはUTC基準
        df['visit_datetime'] = df['visit_time'].apply(
            lambda x: datetime(1601, 1, 1) + timedelta(microseconds=x)
        )
        # タイムゾーンをUTCに設定
        df['visit_datetime'] = df['visit_datetime'].dt.tz_localize('UTC')
        # ローカルタイムゾーンに変換
        df['visit_datetime'] = df['visit_datetime'].dt.tz_convert(local_timezone)

        # ドメインの抽出
        df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)

        # 当日のデータのみを抽出
        target_day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        target_day_end = target_day_start + timedelta(days=1)
        df_target_day = df[(df['visit_datetime'] >= target_day_start) & (df['visit_datetime'] < target_day_end)]

        # 総アクセス数
        total_visits = len(df_target_day)

        if total_visits == 0:
            print("対象日のデータがありません。")
            return total_visits, None, None, None, None, None

        # セッションの計算と滞在時間の計算
        df_target_day = df_target_day.sort_values('visit_time')

        # タイムスタンプを秒単位に変換
        df_target_day['visit_time_sec'] = df_target_day['visit_time'] / 1e6

        # 前回の訪問との時間差を計算
        df_target_day['time_since_last_visit'] = df_target_day['visit_time_sec'] - df_target_day['visit_time_sec'].shift(1)

        # セッションの間隔（秒）
        SESSION_TIMEOUT = config.get('session_timeout', 1800)  # デフォルトは30分

        # セッションIDを生成
        df_target_day['is_new_session'] = df_target_day['time_since_last_visit'] > SESSION_TIMEOUT
        df_target_day['session_id'] = df_target_day['is_new_session'].cumsum()

        # 各ページの滞在時間を計算
        df_target_day['duration'] = df_target_day.groupby('session_id')['visit_time_sec'].shift(-1) - df_target_day['visit_time_sec']
        df_target_day['duration'] = df_target_day['duration'].fillna(30)  # 最後のページの滞在時間を30秒とする

        # 滞在時間が負の場合は0にする
        df_target_day['duration'] = df_target_day['duration'].clip(lower=0)

        # 最大滞在時間を適用
        MAX_DURATION = config.get('max_duration', 1800)  # デフォルトは30分
        df_target_day['duration'] = df_target_day['duration'].clip(upper=MAX_DURATION)

        # ページごとの滞在時間と訪問回数の集計
        df_page_stats = df_target_day.groupby('url').agg({
            'duration': 'sum',
            'url': 'count',
            'title': 'first',
            'domain': 'first',
        }).rename(columns={'duration': 'total_duration', 'url': 'visit_count'})

        # インデックスをリセットして 'url' を列に戻す
        df_page_stats = df_page_stats.reset_index()

        # 過去のデータから 'first_visit' を取得
        # ターゲット日以前のデータのみを使用
        df_before_target = df[df['visit_datetime'] < target_day_end]

        first_visits = df_before_target.groupby('url')['visit_datetime'].min().reset_index()
        first_visits = first_visits.rename(columns={'visit_datetime': 'first_visit'})

        # 'first_visit' を 'df_page_stats' にマージ
        df_page_stats = df_page_stats.merge(first_visits, on='url', how='left')

        # 当日初めて訪問したページのみを抽出
        df_first_visit = df_page_stats[
            (df_page_stats['first_visit'] >= target_day_start) &
            (df_page_stats['first_visit'] < target_day_end)
        ]

        # 当日初めて訪問したページがない場合の処理
        if df_first_visit.empty:
            print("当日初めて訪問したページがありません。")
            top_pages_info = None
        else:
            # スコアの計算
            # 設定ファイルから重みを取得
            duration_weight = config.get('score_weights', {}).get('duration_weight', 0.7)
            visit_count_weight = config.get('score_weights', {}).get('visit_count_weight', 0.3)
            keyword_weight = config.get('score_weights', {}).get('keyword_weight', 1.0)
            domain_weight = config.get('score_weights', {}).get('domain_weight', 1.0)

            # キーワードリストを取得
            priority_keywords = config.get('priority_keywords', [])
            priority_keywords = [kw.lower() for kw in priority_keywords]  # 小文字に変換

            # 優先ドメインのリストを取得
            priority_domains = config.get('priority_domains', [])
            priority_domains = [domain.lower() for domain in priority_domains]  # 小文字に変換

            # タイトルにキーワードが含まれるかを判定
            def keyword_match_score(title):
                if not title:
                    return 0
                title_lower = title.lower()
                return any(kw in title_lower for kw in priority_keywords)

            df_first_visit['keyword_match'] = df_first_visit['title'].apply(keyword_match_score)

            # ドメインが優先ドメインかを判定
            df_first_visit['domain_match'] = df_first_visit['domain'].apply(lambda d: d.lower() in priority_domains)

            # スコアの計算
            df_first_visit['score'] = (
                df_first_visit['total_duration'] * duration_weight +
                df_first_visit['visit_count'] * visit_count_weight +
                df_first_visit['keyword_match'] * keyword_weight +
                df_first_visit['domain_match'] * domain_weight
            )

            # フィルタリング
            exclude_domains = config.get('exclude_domains', [])
            exclude_urls = config.get('exclude_urls', [])
            exclude_url_patterns = config.get('exclude_url_patterns', [])

            # ドメインの除外
            df_first_visit = df_first_visit[~df_first_visit['domain'].isin(exclude_domains)]

            # URLの完全一致による除外
            df_first_visit = df_first_visit[~df_first_visit['url'].isin(exclude_urls)]

            # URLパターンによる除外
            for pattern in exclude_url_patterns:
                df_first_visit = df_first_visit[~df_first_visit['url'].str.match(pattern)]

            # スコア順に並べ替え
            df_first_visit = df_first_visit.sort_values(by='score', ascending=False)

            # === ここで優先ドメインと非優先ドメインに分割 ===
            priority_df = df_first_visit[df_first_visit['domain'].str.lower().isin(priority_domains)]
            non_priority_df = df_first_visit[~df_first_visit['domain'].str.lower().isin(priority_domains)]

            # 非優先ドメインのページで、ドメイン重複を除外
            non_priority_df = non_priority_df.drop_duplicates(subset='domain', keep='first')

            # 優先ドメインのページと非優先ドメインのページを再結合
            df_first_visit = pd.concat([priority_df, non_priority_df], ignore_index=True)

            # 再度スコア順に並べ替え
            df_first_visit = df_first_visit.sort_values(by='score', ascending=False)

            # 上位10件を取得
            top_pages_info = df_first_visit.head(10)[['title', 'url']].reset_index(drop=True)

        # 最も訪問したドメインの計算（当日のデータのみ）
        domain_time = df_target_day.groupby('domain')['duration'].sum()
        domain_visits = df_target_day['domain'].value_counts()
        top_domains = domain_time.sort_values(ascending=False).head(10)
        top_domains_visits = domain_visits.loc[top_domains.index]

        # 時間帯ごとのアクセス数（当日のデータのみ）
        hourly_visits = get_hourly_activity(df_target_day)

        # キーワードの抽出（当日のデータのみ）
        keyword_counts = extract_keywords(df_target_day)

        return total_visits, top_domains, top_domains_visits, top_pages_info, hourly_visits, keyword_counts


def get_hourly_activity(df):
    """
    時間帯ごとのアクセス数を集計する。
    """
    df['hour'] = df['visit_datetime'].dt.hour
    hourly_visits = df.groupby('hour').size()
    return hourly_visits

def extract_keywords(df):
    """
    タイトルからキーワードを抽出し、頻度を集計する。
    """
    tokenizer = Tokenizer()
    # ストップワードのリスト
    stop_words = set(['こと', 'もの', 'これ', 'それ', 'ため', 'さん', 'そう', 'よう', 'とき', 'ところ'])
    all_keywords = []
    for title in df['title'].dropna():
        tokens = tokenizer.tokenize(title)
        for token in tokens:
            # 基本形を取得
            word = token.base_form
            # 品詞を取得
            part_of_speech = token.part_of_speech
            # 品詞が名詞で、4文字以上の単語のみ対象とする
            if part_of_speech.startswith('名詞') and len(word) >= 4:
                # 記号、数字、アルファベットのみの単語を除外
                if not word.isdigit() and not is_punctuation(word) and not is_alphabet(word):
                    if word not in stop_words:
                        all_keywords.append(word)
    keyword_counts = Counter(all_keywords).most_common(10)
    return keyword_counts

def is_punctuation(word):
    """
    単語が記号のみで構成されているかを判定する。
    """
    import string
    return all(char in string.punctuation for char in word)

def is_alphabet(word):
    """
    単語がアルファベットのみで構成されているかを判定する。
    """
    return all('a' <= char.lower() <= 'z' for char in word)

def generate_markdown_text(date_str, total_visits, top_domains, top_domains_visits, top_pages_info, hourly_visits, keyword_counts, md_file_path, config):
    """
    Markdownテキストを生成する。
    """
    # 分析オプションを取得
    analysis_options = config.get('analysis_options', {})
    
    # ファイルの書き込みモードを設定
    if os.path.exists(md_file_path):
        # ファイルが存在する場合は追記モード
        file_mode = 'a'
        # 内容を区切るための区切り線を追加
        md_text = '\n\n---\n\n'
    else:
        # ファイルが存在しない場合は新規作成
        file_mode = 'w'
        md_text = ''
    # Markdownテキストの作成
    md_text += f'# {date_str} のブラウジング要約\n\n'
    
    # 総アクセス数
    if analysis_options.get('total_visits', True) and total_visits is not None:
        md_text += f'- **総アクセス数**: {total_visits}回\n'
    
    if total_visits > 0:
        # 最も訪問したドメイン
        if analysis_options.get('top_domains', True) and top_domains is not None:
            md_text += '\n## 最も訪問したドメイン（滞在時間順）\n\n'
            # テーブルのヘッダーを追加
            md_text += '| ドメイン | 訪問回数 | 滞在時間 |\n'
            md_text += '|---|---|---|\n'
            for domain in top_domains.index:
                time_spent = top_domains[domain]
                visits = top_domains_visits[domain]
                minutes, seconds = divmod(time_spent, 60)
                time_str = f'{int(minutes)}分{int(seconds)}秒'
                md_text += f'| {domain} | {visits}回 | {time_str} |\n'
    
        # 注目したページ
        if analysis_options.get('highlighted_pages', True) and top_pages_info is not None:
            md_text += '\n## 注目したページ\n\n'
            for idx, row in top_pages_info.iterrows():
                title = row['title'] if row['title'] else '（タイトルなし）'
                url = row['url']
                md_text += f'- [{title}]({url})\n'
    
        # 時間帯ごとのアクセス数
        if analysis_options.get('hourly_visits', True) and hourly_visits is not None:
            md_text += '\n## 時間帯ごとのアクセス数\n\n'
            md_text += generate_chartsview_data(hourly_visits)
    
        # キーワード上位10
        if analysis_options.get('top_keywords', True) and keyword_counts is not None:
            md_text += '\n## キーワード上位10\n\n'
            # テーブルのヘッダーを追加
            md_text += '| キーワード | 出現回数 |\n'
            md_text += '|---|---|\n'
            for word, count in keyword_counts:
                md_text += f'| {word} | {count}回 |\n'
    else:
        md_text += '- **データがありません**\n'
    
    # 必要に応じてメモや感想を追加するためのテンプレート
    # md_text += '\n## メモと感想\n\n'
    # md_text += '- '
    return md_text, file_mode


def generate_chartsview_data(hourly_visits):
    """
    hourly_visits を chartsview プラグインのデータ形式に変換する
    """
    chartsview_text = '```chartsview\n'
    chartsview_text += 'type: Column\n'
    chartsview_text += 'options:\n'
    chartsview_text += '  xField: x\n'
    chartsview_text += '  yField: y\n'
    chartsview_text += 'data:\n'
    for hour in range(24):
        count = int(hourly_visits.get(hour, 0))
        chartsview_text += f'  - x: "{hour}時"\n'
        chartsview_text += f'    y: {count}\n'
    chartsview_text += '```\n'
    return chartsview_text


def write_markdown_file(md_text, md_file_path, file_mode):
    """
    Markdownファイルに書き込む。
    """
    try:
        with open(md_file_path, file_mode, encoding='utf-8') as f:
            f.write(md_text)
        print(f'Markdownファイルが {md_file_path} に保存されました。')
    except Exception as e:
        print(f'ファイルの書き込み中にエラーが発生しました: {e}')

def main():
    # 設定ファイルを読み込む
    config = load_config()
    
    # 一時的な履歴ファイルをコピー
    temp_history_path = copy_history_file()
    
    # タイムゾーンの取得
    local_timezone = datetime.now().astimezone().tzinfo
    
    # 対象の日付を取得
    target_date = get_target_date(config.get('date', 'today'), local_timezone)
    
    # 過去の開始タイムスタンプを計算
    past_days = config.get('past_days', 7)  # デフォルトは7日間
    past_date = target_date - timedelta(days=past_days)
    
    # epoch_start を定義（タイムゾーンを統一）
    epoch_start = datetime(1601, 1, 1, tzinfo=local_timezone)
    
    # タイムスタンプを計算
    date_format = config.get('date_format', '%Y_%m_%d')
    date_str = target_date.strftime(date_format)
    
def main():
    # 設定ファイルを読み込む
    config = load_config()

    # 一時的な履歴ファイルをコピー
    temp_history_path = copy_history_file()

    # タイムゾーンの取得
    local_timezone = datetime.now().astimezone().tzinfo

    # 対象の日付を取得
    target_date = get_target_date(config.get('date', 'today'), local_timezone)

    # タイムスタンプを計算
    # ChromeのタイムスタンプはUTC基準
    target_day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    target_day_end = target_day_start + timedelta(days=1)

    # タイムスタンプをUTCに変換
    epoch_start = datetime(1601, 1, 1, tzinfo=timezone.utc)
    micros_start = int((target_day_start.astimezone(timezone.utc) - epoch_start).total_seconds() * 1e6)
    micros_end = int((target_day_end.astimezone(timezone.utc) - epoch_start).total_seconds() * 1e6)

    # 履歴データを取得
    df = fetch_history_data(temp_history_path, micros_start, micros_end)

    # 一時ファイルの削除
    os.remove(temp_history_path)

    # データを加工・分析
    total_visits, top_domains, top_domains_visits, top_pages_info, hourly_visits, keyword_counts = process_history_data(df, target_date, config)

    # 現在の.pyファイルのディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Markdownファイルのパスを設定
    output_dir = config.get('output_dir', os.path.join(current_dir, 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    date_format = config.get('date_format', '%Y_%m_%d')
    date_str = target_date.strftime(date_format)
    md_file_path = os.path.join(output_dir, f'{date_str}.md')

    # Markdownテキストを生成
    md_text, file_mode = generate_markdown_text(
        date_str, total_visits, top_domains, top_domains_visits,
        top_pages_info, hourly_visits, keyword_counts, md_file_path, config
    )

    # Markdownファイルに書き込む
    write_markdown_file(md_text, md_file_path, file_mode)


if __name__ == '__main__':
    main()