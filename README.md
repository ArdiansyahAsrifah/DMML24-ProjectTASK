# Cancer Prediction Risk


## Kelompok

Kelompok 1
- Muhammad Ardiansyah Asrifah - H071221068
- Zabryna Andiny - H071221066
- Izzata Clarissa Salsabila - H071221065
- Elva Aprili Timang - H071221076

## Structure

📦DMML24-ProjectTASK
 ┣ 📂.git
 ┃ ┣ 📂hooks
 ┃ ┃ ┣ 📜applypatch-msg.sample
 ┃ ┃ ┣ 📜commit-msg.sample
 ┃ ┃ ┣ 📜fsmonitor-watchman.sample
 ┃ ┃ ┣ 📜post-update.sample
 ┃ ┃ ┣ 📜pre-applypatch.sample
 ┃ ┃ ┣ 📜pre-commit.sample
 ┃ ┃ ┣ 📜pre-merge-commit.sample
 ┃ ┃ ┣ 📜pre-push.sample
 ┃ ┃ ┣ 📜pre-rebase.sample
 ┃ ┃ ┣ 📜pre-receive.sample
 ┃ ┃ ┣ 📜prepare-commit-msg.sample
 ┃ ┃ ┣ 📜push-to-checkout.sample
 ┃ ┃ ┗ 📜update.sample
 ┃ ┣ 📂info
 ┃ ┃ ┗ 📜exclude
 ┃ ┣ 📂logs
 ┃ ┃ ┣ 📂refs
 ┃ ┃ ┃ ┣ 📂heads
 ┃ ┃ ┃ ┃ ┗ 📜main
 ┃ ┃ ┃ ┗ 📂remotes
 ┃ ┃ ┃ ┃ ┗ 📂origin
 ┃ ┃ ┃ ┃ ┃ ┣ 📜HEAD
 ┃ ┃ ┃ ┃ ┃ ┗ 📜main
 ┃ ┃ ┗ 📜HEAD
 ┃ ┣ 📂objects
 ┃ ┃ ┣ 📂1a
 ┃ ┃ ┃ ┗ 📜70f69cbc9ca7cc9cb2d3d246e62499fba87d42
 ┃ ┃ ┣ 📂21
 ┃ ┃ ┃ ┗ 📜6cad401a32a09607db0733bc11d4a3bb910391
 ┃ ┃ ┣ 📂28
 ┃ ┃ ┃ ┗ 📜465a5f493340e8a950a32733dc5f830e49e9e1
 ┃ ┃ ┣ 📂57
 ┃ ┃ ┃ ┗ 📜dd7526ecbaccdcf2e7744fd76ca887ba32eb93
 ┃ ┃ ┣ 📂94
 ┃ ┃ ┃ ┗ 📜8dd3ffe630af8a14a4d937286b4e8c68ee8316
 ┃ ┃ ┣ 📂95
 ┃ ┃ ┃ ┗ 📜de63d753af010d79a7a7124e19b7c3c9a52bab
 ┃ ┃ ┣ 📂b4
 ┃ ┃ ┃ ┗ 📜52ead5c2bbfb77fbcc32e3181e944d83572aa0
 ┃ ┃ ┣ 📂b6
 ┃ ┃ ┃ ┗ 📜97ec0de9f83af218e569fbb23d1db7519b910d
 ┃ ┃ ┣ 📂c8
 ┃ ┃ ┃ ┗ 📜e64e976092368021718d33c5f2e82b27d5f86d
 ┃ ┃ ┣ 📂c9
 ┃ ┃ ┃ ┗ 📜118c3502c0604039b4cb8194cab1f1f10cafed
 ┃ ┃ ┣ 📂fa
 ┃ ┃ ┃ ┗ 📜f0b110e933d13b34ed251e3719debf851c7960
 ┃ ┃ ┣ 📂fb
 ┃ ┃ ┃ ┗ 📜10d3ed96b06c1c56ae6a257e32dbc67d368c27
 ┃ ┃ ┣ 📂info
 ┃ ┃ ┗ 📂pack
 ┃ ┃ ┃ ┣ 📜pack-aad2fb81988c356f90fa0cd026a301cc10adbe12.idx
 ┃ ┃ ┃ ┗ 📜pack-aad2fb81988c356f90fa0cd026a301cc10adbe12.pack
 ┃ ┣ 📂refs
 ┃ ┃ ┣ 📂heads
 ┃ ┃ ┃ ┗ 📜main
 ┃ ┃ ┣ 📂remotes
 ┃ ┃ ┃ ┗ 📂origin
 ┃ ┃ ┃ ┃ ┣ 📜HEAD
 ┃ ┃ ┃ ┃ ┗ 📜main
 ┃ ┃ ┗ 📂tags
 ┃ ┣ 📜COMMIT_EDITMSG
 ┃ ┣ 📜config
 ┃ ┣ 📜description
 ┃ ┣ 📜FETCH_HEAD
 ┃ ┣ 📜HEAD
 ┃ ┣ 📜index
 ┃ ┣ 📜ORIG_HEAD
 ┃ ┗ 📜packed-refs
 ┣ 📂assets
 ┣ 📂datasets
 ┃ ┗ 📜cancer_risk_data.csv
 ┣ 📂src
 ┃ ┣ 📜logistic_regression_model.pkl
 ┃ ┣ 📜scaler.pkl
 ┃ ┗ 📜train_model.py
 ┣ 📜app.py
 ┗ 📜README.md