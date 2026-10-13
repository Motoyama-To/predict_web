// ======================================================
// HTML要素の取得
// ======================================================

// 画像をドラッグ＆ドロップする領域
const dropZone = document.getElementById('dropZone');

// ファイル選択要素
const fileInput = document.getElementById('fileInput');

// 選択した画像のプレビュー
const preview = document.getElementById('preview');

// AI解析開始ボタン
const btn = document.getElementById('predictBtn');

// 「解析中」「解析完了」などの表示
const statusEl = document.getElementById('status');

// AI解析結果全体
const results = document.getElementById('results');

// AI解析結果を挿入する場所
const content = document.getElementById('resultContent');


// ======================================================
// 現在選択されている画像
// ======================================================

let selectedFile = null;


// ======================================================
// HTMLエスケープ
// ======================================================

// HTMLに文字列を安全に表示するための関数
const esc = s =>
    String(s ?? '').replace(
        /[&<>'"]/g,
        c => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[c])
    );


// ======================================================
// 選択された画像を設定
// ======================================================

function setFile(file) {

    // ファイルが存在しない場合
    if (!file) return;

    // 画像ファイル以外の場合
    if (!file.type.startsWith('image/')) return;

    // 選択されたファイルを保存
    selectedFile = file;

    // プレビュー用URLを作成
    preview.src = URL.createObjectURL(file);

    // 画像表示状態へ変更
    dropZone.classList.add('has-image');

    // 判定ボタンを有効化
    btn.disabled = false;

    // 前回の解析結果を非表示
    results.classList.add('hidden');

    // ステータスを空にする
    statusEl.textContent = '';
}


// ======================================================
// クリックによる画像選択
// ======================================================

dropZone.onclick = () => {
    fileInput.click();
};


// ======================================================
// ファイル選択ダイアログ
// ======================================================

fileInput.onchange = e => {

    const file = e.target.files[0];

    setFile(file);
};


// ======================================================
// ドラッグ中の処理
// ======================================================

['dragenter', 'dragover'].forEach(eventName => {

    dropZone.addEventListener(eventName, e => {

        // ブラウザ標準処理を停止
        e.preventDefault();

        // ドラッグ中の表示
        dropZone.classList.add('drag');
    });
});


// ======================================================
// ドラッグ領域から離れたとき・ドロップしたとき
// ======================================================

['dragleave', 'drop'].forEach(eventName => {

    dropZone.addEventListener(eventName, e => {

        // ブラウザ標準処理を停止
        e.preventDefault();

        // ドラッグ中表示を解除
        dropZone.classList.remove('drag');
    });
});


// ======================================================
// 画像をドロップしたとき
// ======================================================

dropZone.addEventListener('drop', e => {

    const file = e.dataTransfer.files[0];

    setFile(file);
});


// ======================================================
// Ctrl + V で画像を貼り付け
// ======================================================

document.addEventListener('paste', e => {

    // クリップボードのデータを確認
    for (const item of e.clipboardData.items) {

        // 画像の場合
        if (item.type.startsWith('image/')) {

            const file = item.getAsFile();

            setFile(file);

            // 最初の画像だけ取得
            break;
        }
    }
});


// ======================================================
// Top 5 品種予測の棒グラフ
// ======================================================

function bars(top5) {

    return top5.map(x => `

        <!-- 1つの品種の予測結果 -->
        <div class="bar-row">

            <!-- 品種名 -->
            <b>
                ${esc(x.breed)}
            </b>

            <!-- 棒グラフ -->
            <div class="bar-track">

                <div
                    class="bar"
                    style="width:${Math.max(
                        0,
                        Math.min(100, Number(x.probability))
                    )}%"
                ></div>

            </div>

            <!-- 確率 -->
            <span>
                ${Number(x.probability).toFixed(2)}%
            </span>

        </div>

    `).join('');
}


// ======================================================
// 類似画像検索結果
// ======================================================

function similar(items) {

    return items.map((x, i) => `

        <!-- 1枚の類似画像 -->
        <div class="similar-item">

            <!-- 類似画像 -->
            <img
                src="${x.image}"
                alt="類似画像${i + 1}"
            >

            <!-- 類似画像の情報 -->
            <div class="similar-meta">

                <!-- 品種名 -->
                <b>
                    ${esc(x.breed)}
                </b>

                <!-- 類似度 -->
                類似度 ${Number(x.similarity).toFixed(3)}

            </div>

        </div>

    `).join('');
}


// ======================================================
// 1つのAI解析結果を作成
// ======================================================

function resultBlock(r, crop, index, total) {

    return `

        <!-- ========================================== -->
        <!-- メインのAI解析結果 -->
        <!-- ========================================== -->

        <div class="result-card hero-result">

            <!-- YOLOによって切り抜いた画像 -->
            <img
                src="${crop}"
                alt="YOLO切り抜き画像"
            >

            <!-- AI判定結果 -->
            <div>

                <!-- 動物の種類 -->
                <div class="animal">

                    ${esc(r.animal_type)}

                    ${
                        total > 1
                            ? `・候補 ${index + 1}`
                            : ''
                    }

                </div>

                <!-- 品種 -->
                <div class="breed">
                    ${esc(r.breed)}
                </div>

                <!-- 推定確率 -->
                <div class="confidence">
                    推定確率 ${Number(r.confidence).toFixed(2)}%
                </div>

            </div>

        </div>


        <!-- ========================================== -->
        <!-- Top 5 と Grad-CAM -->
        <!-- ========================================== -->

        <div class="summary-grid">


            <!-- Top 5 -->
            <div class="result-card">

                <h3>
                    Top 5 品種予測
                </h3>

                ${bars(r.top5)}

            </div>


            <!-- Grad-CAM -->
            <div class="result-card">

                <h3>
                    Grad-CAM
                </h3>

                <!--
                    注意：
                    imgタグの開始タグ内部には
                    HTMLコメントを書かない
                -->

                <img
                    src="${r.gradcam}"
                    alt="Grad-CAM"
                    style="width:100%;border-radius:14px"
                >

            </div>


        </div>


        <!-- ========================================== -->
        <!-- 類似画像検索 -->
        <!-- ========================================== -->

        <div class="result-card">

            <h3>
                類似画像検索
            </h3>

            <div class="similar-grid">

                ${similar(r.similar_images)}

            </div>

        </div>


        <!-- ========================================== -->
        <!-- 品種説明 -->
        <!-- ========================================== -->

        <div class="result-card">

            <h3>
                ${esc(r.breed)} について
            </h3>

            <div class="description">
                ${esc(r.description)}
            </div>

        </div>

    `;
}


// ======================================================
// 「この画像を判定する」ボタン
// ======================================================

btn.onclick = async () => {

    // 画像が選択されていなければ終了
    if (!selectedFile) return;


    // ==================================================
    // 解析開始
    // ==================================================

    // 二重クリック防止
    btn.disabled = true;

    // ステータス表示
    statusEl.textContent = 'AIが解析しています…';

    // 前回の結果を削除
    content.innerHTML = '';

    // 結果領域を非表示
    results.classList.add('hidden');


    // ==================================================
    // サーバーへ送信するデータ
    // ==================================================

    const fd = new FormData();

    // Python側の request.files['image'] に対応
    fd.append(
        'image',
        selectedFile
    );


    // ==================================================
    // Flask APIへ送信
    // ==================================================

    try {

        const res = await fetch(
            '/api/predict',
            {
                method: 'POST',
                body: fd
            }
        );


        // ==================================================
        // JSONデータを取得
        // ==================================================

        const data = await res.json();


        // ==================================================
        // HTTPエラー確認
        // ==================================================

        if (!res.ok) {

            throw new Error(
                data.error ||
                '解析に失敗しました'
            );
        }


        // ==================================================
        // 返却データの確認
        // ==================================================

        if (!data.results) {

            throw new Error(
                '解析結果を取得できませんでした。'
            );
        }


        // ==================================================
        // YOLO判定結果
        // ==================================================

        const detectionCount =
            Array.isArray(data.detections)
                ? data.detections.length
                : 0;


        content.innerHTML = `

            <!-- YOLOによる物体検出結果 -->
            <div class="result-card">

                <b>
                    YOLO判定：
                </b>

                ${esc(data.detected_type)}

                <span style="color:#687386">
                    検出数 ${detectionCount}
                </span>

            </div>

        `;


        // ==================================================
        // AI品種分類結果を追加
        // ==================================================

        content.innerHTML += data.results.map(

            (r, i) => resultBlock(

                r,

                // YOLOで切り抜いた画像
                data.cropped_image,

                // 現在の候補番号
                i,

                // 候補総数
                data.results.length
            )

        ).join('');


        // ==================================================
        // 結果を表示
        // ==================================================

        results.classList.remove('hidden');


        // ==================================================
        // 完了表示
        // ==================================================

        statusEl.textContent =
            '解析が完了しました。';


        // ==================================================
        // 解析結果までスクロール
        // ==================================================

        results.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });

    }

    // ======================================================
    // エラー発生時
    // ======================================================

    catch (e) {

        statusEl.innerHTML = `

            <span class="error">
                ${esc(e.message)}
            </span>

        `;

    }

    // ======================================================
    // 最後に必ず実行
    // ======================================================

    finally {

        // 判定ボタンを再び有効化
        btn.disabled = false;
    }
};
