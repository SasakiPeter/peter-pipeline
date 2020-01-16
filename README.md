# 機械学習パイプライン

私が普段使っているパイプラインです。

## 環境構築

docker で仮想環境を構築しています。次のコマンドで、環境が作成されます。

```shell
$ docker-compose up -d
```

## フォルダ構成

configフォルダ内の設定ファイルに基づいて、学習が行われます。

```
|- config/
	|- settings.py        - 設定ファイル
|- data/
	|- train.csv
	...
|- docker/
	|- Dockerfile
	|- requirements.txt
|- models/
	|- ...                - 学習済みモデル
|- pipeline/
	|- conf/
	|- core/
		|- management/
			|- commands/
				|- ...    - 実行コマンド
	|- predict/           - 予測モジュール
	|- preprocess/        - 前処理モジュール
	|- train/             - 学習モジュール
|- submits/
	|- ...                - 予測ファイル
```

## 実行

前処理、学習、予測までの全てのプロセスが実行されます。

```shell
$ docker-compose exec python manage.py runall
```

前処理のみ実行されます。

```shell
$ docker-compose exec python manage.py runpreprocess
```

学習のみ実行されます。

```shell
$ docker-compose exec python manage.py runtrain
```

予測のみ実行されます。

```shell
$ docker-compose exec python manage.py runtrain
```