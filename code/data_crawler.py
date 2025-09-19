import os
import requests
import cv2
import numpy as np
import time
import random
from tqdm import tqdm
import json
from urllib.parse import quote
import hashlib


class BilibiliCrawler:
    def __init__(self, save_dir="dataset/raw1"):
        self.save_dir = save_dir
        self.session = requests.Session()
        self._init_headers()
        self._init_cookies()
        os.makedirs(save_dir, exist_ok=True)

    def _init_headers(self):
        """初始化请求头"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
            'Origin': 'https://www.bilibili.com',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'sec-ch-ua': '"Chromium";v="91", " Not;A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        }
        self.session.headers.update(self.headers)

    def _init_cookies(self):
        """初始化Cookies"""
        self.cookies = {
            'SESSDATA': os.getenv('BILIBILI_SESSDATA', ''),
            'bili_jct': os.getenv('BILIBILI_JCT', ''),
            'DedeUserID': os.getenv('BILIBILI_USERID', ''),
            'sid': hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        }
        self.session.cookies.update(self.cookies)

    def _make_request(self, url, max_retries=3):
        """封装请求方法，包含重试机制"""
        for attempt in range(max_retries):
            try:
                # 每次请求前随机延时
                time.sleep(random.uniform(1, 3))

                response = self.session.get(url, timeout=15)

                # 检查HTTP状态码
                if response.status_code == 412:
                    print(f"请求被拒绝，尝试更换Cookies或User-Agent (尝试 {attempt + 1}/{max_retries})")
                    self._rotate_user_agent()
                    continue

                response.raise_for_status()

                # 检查API返回状态码
                data = response.json()
                if data.get('code') != 0:
                    print(f"B站API返回错误: {data.get('message')}")
                    return None

                return data

            except Exception as e:
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(random.uniform(5, 10))

        return None

    def _rotate_user_agent(self):
        """轮换User-Agent"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        new_agent = random.choice(user_agents)
        self.session.headers['User-Agent'] = new_agent

    def download_from_bilibili(self, query, num_videos=10):
        """从B站搜索并下载视频片段"""
        fight_dir = os.path.join(self.save_dir, "fight")
        nofight_dir = os.path.join(self.save_dir, "nofight")
        os.makedirs(fight_dir, exist_ok=True)
        os.makedirs(nofight_dir, exist_ok=True)

        # 使用备用搜索策略
        fight_query = f"{query} 打架"
        fight_videos = self._search_with_retry(fight_query, num_videos)
        if fight_videos:
            self._download_videos(fight_videos, fight_dir, "fight")

        nofight_query = f"{query} 日常"
        nofight_videos = self._search_with_retry(nofight_query, num_videos)
        if nofight_videos:
            self._download_videos(nofight_videos, nofight_dir, "normal")

    def _search_with_retry(self, query, num_videos, max_retries=3):
        """带重试的搜索方法"""
        for attempt in range(max_retries):
            videos = self._search_bilibili(query, num_videos)
            if videos:
                return videos
            print(f"搜索失败，尝试 {attempt + 1}/{max_retries}")
            time.sleep(random.uniform(10, 20))
        return []

    def _search_bilibili(self, query, num_videos):
        """搜索B站视频"""
        # 尝试两种搜索API
        search_urls = [
            f"https://api.bilibili.com/x/web-interface/search/type?keyword={quote(query)}&search_type=video",
            f"https://api.bilibili.com/x/web-interface/wbi/search/type?keyword={quote(query)}&search_type=video"
        ]

        for url in search_urls:
            data = self._make_request(url)
            if data and 'data' in data and 'result' in data['data']:
                videos = []
                for video in data['data']['result']:
                    videos.append({
                        'bvid': video['bvid'],
                        'title': video['title'],
                        'duration': video['duration']
                    })
                    if len(videos) >= num_videos:
                        return videos
                return videos

        return []

    def _get_video_info(self, bvid):
        """获取视频详细信息"""
        info_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        data = self._make_request(info_url)
        if data and 'data' in data:
            return {
                'cid': data['data']['cid'],
                'owner': data['data']['owner']['name'],
                'pubdate': data['data']['pubdate']
            }
        return None

    def _download_videos(self, videos, save_dir, prefix):
        """下载B站视频"""
        for i, video in enumerate(videos):
            try:
                video_info = self._get_video_info(video['bvid'])
                if not video_info:
                    print(f"无法获取视频 {video['bvid']} 的信息")
                    continue

                # 尝试多种方式获取视频地址
                play_urls = [
                    f"https://api.bilibili.com/x/player/playurl?bvid={video['bvid']}&cid={video_info['cid']}&qn=16&fnval=16",
                    f"https://api.bilibili.com/x/player/wbi/playurl?bvid={video['bvid']}&cid={video_info['cid']}&qn=16&fnval=16"
                ]

                play_data = None
                for url in play_urls:
                    play_data = self._make_request(url)
                    if play_data:
                        break

                if not play_data:
                    print(f"无法获取视频 {video['bvid']} 的播放地址")
                    continue

                # 解析视频地址
                video_url = None
                if 'durl' in play_data['data']:
                    video_url = play_data['data']['durl'][0]['url']
                elif 'dash' in play_data['data']:
                    video_url = play_data['data']['dash']['video'][0]['baseUrl']

                if not video_url:
                    print(f"无法解析视频 {video['bvid']} 的播放地址")
                    continue

                # 下载视频
                video_path = os.path.join(save_dir, f"{prefix}_{i}.mp4")
                self._download_file(video_url, video_path, video['bvid'])
                print(f"已下载视频 {i + 1}/{len(videos)}: {video['title']}")

                time.sleep(random.uniform(15, 30))  # 更长的随机延时

            except Exception as e:
                print(f"下载视频 {video['bvid']} 时出错: {str(e)[:100]}")
                continue

    def _download_file(self, url, save_path, bvid):
        """下载文件并保存到本地"""
        headers = self.headers.copy()
        headers['Referer'] = f"https://www.bilibili.com/video/{bvid}"

        try:
            with self.session.get(url, headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))

                # 使用进度条
                progress = tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"下载 {os.path.basename(save_path)}"
                )

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))

                progress.close()

        except Exception as e:
            print(f"下载文件时出错: {str(e)[:100]}")
            if os.path.exists(save_path):
                os.remove(save_path)
            raise

    # 保留原有的帧提取和处理方法
    def extract_frames(self, video_path, output_dir, fps=5):
        """从视频中提取帧"""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        save_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
                frame_path = os.path.join(output_dir, f"frame_{save_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                save_count += 1

            frame_count += 1

        cap.release()
        print(f"从 {video_path} 中提取了 {save_count} 帧")

    def process_raw_data(self, output_dir="dataset/processed1"):
        """处理原始视频数据，提取帧并划分训练/验证集"""
        raw_dir = self.save_dir
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")

        # 创建输出目录
        os.makedirs(os.path.join(train_dir, "fight"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "nofight"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "fight"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "nofight"), exist_ok=True)

        # 处理打架视频
        fight_videos = os.listdir(os.path.join(raw_dir, "fight"))
        for i, video in enumerate(tqdm(fight_videos)):
            video_path = os.path.join(raw_dir, "fight", video)
            if i < len(fight_videos) * 0.8:  # 80%用于训练
                output_subdir = os.path.join(train_dir, "fight", video.split(".")[0])
            else:  # 20%用于验证
                output_subdir = os.path.join(val_dir, "fight", video.split(".")[0])

            self.extract_frames(video_path, output_subdir)

        # 处理正常视频
        nofight_videos = os.listdir(os.path.join(raw_dir, "nofight"))
        for i, video in enumerate(tqdm(nofight_videos)):
            video_path = os.path.join(raw_dir, "nofight", video)
            if i < len(nofight_videos) * 0.8:  # 80%用于训练
                output_subdir = os.path.join(train_dir, "nofight", video.split(".")[0])
            else:  # 20%用于验证
                output_subdir = os.path.join(val_dir, "nofight", video.split(".")[0])

            self.extract_frames(video_path, output_subdir)


if __name__ == "__main__":
    crawler = BilibiliCrawler()
    crawler.download_from_bilibili("街头监控", num_videos=10)
    crawler.process_raw_data()