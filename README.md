<h2 data-ke-size="size26"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b><b>DFL(DeepFaceLab)실행 환경 설정</b></b></span></h2>
<p><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">DFL은 중국에서 개발 및 배포된 소스로, 소스 영상에서 얼굴을 추출하여 타겟 영상에 합성, 손쉽게 Deep fake를 만들 수 있음<br>사진/영상 포함 포스팅 : https://wonder-j.tistory.com/14?category=778607 </span></p>
<p><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">출처 : <a href="https://github.com/dream80/DeepFaceLab_Linux" target="_blank" rel="noopener">DeepFaceLab</a></span></p>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b><a href="https://m.blog.naver.com/PostView.nhn?blogId=dsz08082&amp;logNo=221185332846&amp;proxyReferer=https:%2F%2Fwww.google.com%2F" target="_blank" rel="noopener">아나콘다 설치</a></b></span></h3>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b>DFL 가상환경 만들기</b></span></h3>
<pre id="code_1589272965046" class="python" data-ke-language="python" data-ke-type="codeblock"><code>conda create -y -n deepfacelab python=3.6.6 cudatoolkit=9.0 cudnn=7.3.1
conda activate deepfacelab</code></pre>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b>clone <b>DFL</b></b></span></h3>
<pre id="code_1589273006282" class="python" data-ke-language="python" data-ke-type="codeblock"><code>git clone https://github.com/dream80/DeepFaceLab_Linux.git</code></pre>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b>Requirement 설치</b></span></h3>
<pre id="code_1589273024994" class="python" data-ke-language="python" data-ke-type="codeblock"><code>cd DeepFaceLab_Linux
python -m pip install -r requirements-cuda.txt</code></pre>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b>ffmpeg 설치(conda)</b></span></h3>
<pre id="code_1589273036370" class="python" data-ke-language="python" data-ke-type="codeblock"><code>conda install ffmpeg==4.0.2
conda insatll ffmpeg</code></pre>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">Tensor version 맞추기(CUDA 10.0 기준)</span></h3>
<pre id="code_1589273076999" class="python" data-ke-language="python" data-ke-type="codeblock"><code>conda install tensorflow-gpu==1.13.1 # 이때 cudatoolkit, cudnn 등 설치 됨
conda install tensorflow==1.13.1
conda install tensorboard==1.13.1</code></pre>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">workspace에 영상 넣기</span></h3>
<p><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">영상은 자유롭게 넣되, 아래와 같이 naming</span></p>
<ul style="list-style-type: disc;">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">합성하고 싶은 얼굴이 든 영상은 : <a href="https://www.youtube.com/watch?v=3KCaBzeaa9M" target="_blank" rel="noopener">data_src.<span style="color: #333333;">mp4</span></a></span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">합성 대상 영상 : <a href="https://www.youtube.com/watch?v=q5G0wQxLSSo" target="_blank" rel="noopener">data_dst.mp4</a></span></li>
</ul>
<pre id="code_1589274767534" class="python" style="margin: 20px auto 0px; display: block; overflow: auto; padding: 15px; color: #383a42; background: #f6f7f8; font-size: 14px; border-radius: 3px; font-family: Menlo, Consolas, Monaco, monospace; border: 1px solid #dddddd; cursor: default; z-index: 1; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; text-decoration-style: initial; text-decoration-color: initial;" data-ke-language="python" data-ke-type="codeblock"><code>$ scp -P 16022 ./data_src.mp4 {개인서버 개정}@{서버주소}:{DeepFaceLab 경로}/DeepFaceLab_Linux/workspace/
$ scp -P 16022 ./data_dst.mp4 {개인서버 개정}@{서버주소}:{DeepFaceLab 경로}/DeepFaceLab_Linux/workspace/</code></pre>
<p><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR'; color: #ee2323;">본 예시에서는 수지 광고 영상에 아이유의 얼굴을 합성할 것임</span></p>
<p><img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F055MW%2FbtqD7GI751u%2F723kFyBrR0lhgCUn8dXtIk%2Fimg.png">이 아이유를<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdednnP%2FbtqD6QrGnwD%2FxxJLjnrHbHs9HnNjNmXfz1%2Fimg.png">요 수지로 합성</p>
<h2><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><b><b>DFL 실행</b></b></span></h2>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">1. env 실행</span></h3>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">DeepFaceLab_Linux/script/env.sh</span></li>
</ul>
<pre id="code_1589274007083" class="python" data-ke-language="python" data-ke-type="codeblock"><code>export DFL_CUDA="CUDA_VISIBLE_DEVICES=1"
export DFL_PYTHON="python3.6"
export DFL_WORKSPACE="../workspace"

if [ ! -d "$DFL_WORKSPACE" ]; then
    mkdir "$DFL_WORKSPACE"
    mkdir "$DFL_WORKSPACE/data_src"
    mkdir "$DFL_WORKSPACE/data_src/aligned"
    mkdir "$DFL_WORKSPACE/data_src/aligned_debug"
    mkdir "$DFL_WORKSPACE/data_dst"
    mkdir "$DFL_WORKSPACE/data_dst/aligned"
    mkdir "$DFL_WORKSPACE/data_dst/aligned_debug"
    mkdir "$DFL_WORKSPACE/model"
fi

export DFL_SRC="../"</code></pre>
<h3 data-ke-size="size23"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">2. data_src 프레임 추출</span></h3>
<pre id="code_1589274696165" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./2_extract_PNG_from_video_data_src.sh</code></pre>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">3. data_dst 프레임 추출</span></h3>
<pre id="code_1589274726955" class="python" style="display: block; overflow: auto; padding: 15px; color: #383a42; background: #f6f7f8; font-size: 14px; border-radius: 3px; font-family: Menlo, Consolas, Monaco, monospace; border: 1px solid #dddddd; margin: 20px auto 0px; cursor: default; z-index: 1; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; text-decoration-style: initial; text-decoration-color: initial;" data-ke-language="python" data-ke-type="codeblock"><code>$ ./3_extract_PNG_from_video_data_dst.sh</code></pre>
<h4 data-ke-size="size20"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">3-1. data_dst&nbsp; denoise(optional)</span></h4>
<pre id="code_1589274860526" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./3_other_denoise_extracted_data_dst.sh</code></pre>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">4. data_src 프레임에서 face 추출</span></h3>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">face 추출의 경우 S3FD, MT 등 option이 있으나 S3FD권장</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">이 작업은 환경이 CPU일 때 매우 느리게 작동함</span></li>
</ul>
<pre id="code_1589274960391" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./4_data_src_extract_faces_S3FD_best_GPU.sh</code></pre>
<p><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FPwxR3%2FbtqD7ZaBi4O%2FmMYGMK7Hmwx0RrdtLC9Rgk%2Fimg.jpg"><br>이렇게 프레임에서 얼굴만 잘라줌</p>
<h4 data-ke-size="size20"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">4.1. face를 유사 히스토그램으로 정렬</span></h4>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">similar_histogram : 이 옵션은 추출된 face를 유사 히스토그램으로 정렬 해준다</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';"><span style="color: #333333;">소스 영상에 원하는 사람의 얼굴(아이유)이 아닌 타인의 얼굴이 있을 때 제거가 용이하다</span><span style="color: #333333;"></span></span></li>
</ul>
<pre id="code_1589275346064" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./4.2.2_data_src_sort_by_similar_histogram.sh</code></pre>
<h4 data-ke-size="size20"><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">4.1. face를 final option으로 정렬</span></h4>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">final : 권장되는 정렬로, 소스로 활용하기 좋지 않은 프레임을 버리고, 필요한 프레임(yaw 기준으로 다양하게)만 남겨준다</span></li>
</ul>
<pre id="code_1589275524872" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./4.2.6_data_src_sort_by_final.sh</code></pre>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">5. data_dst 프레임에서 face 추출</span></h3>
<pre id="code_1589275982578" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./5_data_dst_extract_faces_S3FD_best_GPU.sh</code></pre>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">6. model train</span></h3>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">model도 여러가지가 있는데, blending 할 때 option이 다양한 SAE로 실행</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">학습이 정상적으로 실행되면 아래와 같은 미리보기 이미지가 workspace/model에 저장됨</span></li>
</ul>
<pre id="code_1589276588960" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./6_train_SAE.sh</code></pre>
<p><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOHjTN%2FbtqD3c4aWb4%2FZAQQ7itguCFlsWVH8nMej0%2Fimg.jpg"><br>iter : 100<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FB0pmU%2FbtqD8hhYgzs%2FVs31gk6Wo5H1BIWHiw1zjK%2Fimg.jpg"><br>iter : 20,000</p>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">7. Convert</span></h3>
<pre id="code_1589347791029" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./7_convert_{사용한 모델명}.sh</code></pre>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">아래와 같이 merged된 프레임을 workspace/data_dst/merged 경로에 얻을 수 있다.</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">convert를 진행하다보면 많은 option이 있는데, 이를 어떻게 조정하느냐에 따라 영상의 자연스러움이 다르다</span></li>
</ul>
<p><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcGZ6hh%2FbtqD4yTGbMo%2Fl0BUsQqCuT8nJ2OxD5aDGK%2Fimg.png">converting option을 기본 값으로 준 것<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fv8gya%2FbtqD8OAi2Uu%2Fx3ur4RTtkNGz0HYsk1d3AK%2Fimg.png">조금 더 자연스럽도록 option을 수정한 것<br>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbm0UBz%2FbtqD6Q6YLtv%2FVGPhkIwW13kHwYjAO8vHr0%2Fimg.png", width = 400>  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbru2VK%2FbtqD7Y4ytko%2FezuYcPcsurXkyEcTgEw5a1%2Fimg.png", width = 400><br>
원본 이미지(좌) 합성된 이미지(우)<br></p>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">위 예시에서 알 수 있듯 완벽히 합성되지 않는 frame 들이 있다(사실상 자연스러운 부분이 더 적음)</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">턱의 검은 건 왜 생긴지 알 수가 없고, 미표한 표정의 변화까지 완벽히 합성되지는 않는 모양</span></li>
</ul>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">8. mp4 영상으로 추출</span></h3>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">추출된 영상은 /workspace 안에 result.mp4 로 저장된다.</span></li>
</ul>
<pre id="code_1589348245613" class="python" data-ke-language="python" data-ke-type="codeblock"><code>$ ./8_converted_to_mp4.sh</code></pre>
<p>&nbsp;</p>
<figure data-ke-type="video" data-ke-style="alignCenter" data-video-host="kakaotv" data-video-url="https://tv.kakao.com/channel/3568028/cliplink/408988123" data-video-thumbnail="https://scrap.kakaocdn.net/dn/tqbyA/hyF2CQOsv0/50kHHFQZbHk4Bu6AAcaU4K/img.png?width=854&amp;height=480&amp;face=286_92_455_277,https://scrap.kakaocdn.net/dn/ZOH7S/hyF1FVTDEc/h2gm9ntIegC4BkQZGMHFuk/img.jpg?width=640&amp;height=360&amp;face=0_0_640_360" data-video-width="854" data-video-height="480" data-ke-mobilestyle="widthContent" data-video-play-service="daum_tistory"><iframe src="https://play-tv.kakao.com/embed/player/cliplink/408988123?service=daum_tistory" width="854" height="480" frameborder="0" allowfullscreen="true"></iframe>
<figcaption>합성된 딥페이크 영상</figcaption>
</figure>
<h3><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">9. review</span></h3>
<ul style="list-style-type: disc;" data-ke-list-type="disc">
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">GPU 환경에서 train에 약 16시간 정도 소요, 가치가 있나..(ㅎㅎ)</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">Youtube 등에서 훨씬 자연스럽게 합성된 deep fake 들에 비해 뭐가 부족한걸까?</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">합성된 영상은 퀄리티를 더 높이려면 어떤 시도를 할 수 있을지?</span></li>
<li><span style="font-family: 'Noto Sans Demilight', 'Noto Sans KR';">특정 헤어스타일이 어울릴지에 대한 의사결정에 좋을 것 같다(?)</span></li>
</ul>
