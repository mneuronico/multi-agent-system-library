# maws — MAS on AWS (Lambda + API Gateway)

**maws** es una librería auxiliar para desplegar bots de la librería **MAS** en **AWS Lambda** detrás de **API Gateway**, con:

* inicialización perezosa del `AgentSystemManager`
* manejo de webhooks para **WhatsApp** y **Telegram**
* verificación de webhook (WhatsApp)
* **auto-invocación** para procesar en segundo plano (evita timeouts de API Gateway)
* **locking** por usuario con **DynamoDB TTL** para evitar concurrencia cruzada
* sincronización de **tokens/keys** entre **S3** y el FS efímero `/tmp`
* carga de configuración desde **SSM Parameter Store** (SecureString)

> Si ya usás MAS, maws te permite reducir tu `lambda_function.py` a \~3 líneas.

---

## Instalación

### pip (desde subdirectorio del repo)

```bash
pip install \
  "git+https://github.com/mneuronico/multi-agent-system-library.git#subdirectory=maws" \
  "git+https://github.com/mneuronico/multi-agent-system-library.git"
```

* La primera línea instala **maws** (subcarpeta `maws`).
* La segunda instala **MAS** (raíz del repo).
* También podés **pinear a un commit/tag**:

```bash
pip install \
  "git+https://github.com/mneuronico/multi-agent-system-library.git@<commit>#subdirectory=maws" \
  "git+https://github.com/mneuronico/multi-agent-system-library.git@<commit>"
```

### `requirements.txt` (ejemplo)

```txt
boto3
requests
git+https://github.com/mneuronico/multi-agent-system-library.git#subdirectory=maws
git+https://github.com/mneuronico/multi-agent-system-library.git
```

> **Notas**
>
> * `boto3` y `botocore` son dependencias de runtime en Lambda; incluir `boto3` en tus deps está bien aunque Lambda lo provea.
> * MAS es dependencia de maws (se importa como `from mas import AgentSystemManager, WhatsappBot, TelegramBot`).

---

## Uso rápido

Tu `lambda_function.py` puede quedar así de simple:

```python
# lambda_function.py
import os
from maws import build_lambda_handler

BOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")  # "whatsapp" | "telegram"
lambda_handler = build_lambda_handler(BOT_TYPE)
```

Eso es todo. `maws` hace el wiring con MAS, maneja GET/POST del webhook, colas por auto-invocación, import/export de historial, locks, etc.

---

## Variables de entorno (contrato)

| Variable                   | Obligatorio | Default      | Descripción                                                        |
| -------------------------- | ----------- | ------------ | ------------------------------------------------------------------ |
| `BOT_TYPE`                 | No          | `"whatsapp"` | `"whatsapp"` o `"telegram"`.                                       |
| `VERBOSE`                  | No          | `"false"`    | `true/false` para logging detallado de MAS y bots.                 |
| `BUCKET_NAME`              | **Sí**      | —            | S3 con historiales y (opcional) tokens/keys.                       |
| `ENV_PARAMETER_NAME`       | No          | —            | Nombre del parámetro **SSM** (SecureString) con `.env` a inyectar. |
| `SYNC_TOKENS_S3`           | No          | `"true"`     | Si está `true`, busca tokens en S3 antes que en el paquete.        |
| `TOKENS_S3_PREFIX`         | No          | `"secrets"`  | Prefijo en S3 para tokens/keys (p. ej. `secrets/<archivo>`).       |
| `SPECIAL_TOKEN_FILES_JSON` | No          | `"[]"`       | JSON con lista de archivos de token a sincronizar hacia `/tmp`.    |
| `TOKEN_ENV_MAP_JSON`       | No          | `"{}"`       | JSON `{ "archivo": "ENV_VAR" }` para exponer rutas en env.         |
| `LOCKS_TABLE_NAME`         | No          | —            | Nombre de tabla DynamoDB para **locking** por usuario.             |
| `LOCK_TTL_SECONDS`         | No          | `"180"`      | TTL del lock (segundos).                                           |

### ¿Cómo funcionan los tokens?

* **SPECIAL\_TOKEN\_FILES\_JSON**: lista de nombres de archivos (p. ej. `["openai.key", "facebook.json"]`).
  maws intentará:

  1. Descargar `s3://BUCKET_NAME/TOKENS_S3_PREFIX/<archivo>` → `/tmp/<archivo>`, o
  2. Copiar `<archivo>` desde el **paquete** (read-only) → `/tmp/<archivo>` (fallback).

* **TOKEN\_ENV\_MAP\_JSON**: mapa para **exponer** el path en env; ej:

  ```json
  { "facebook.json": "FACEBOOK_CREDENTIALS_PATH", "openai.key": "OPENAI_API_KEY_PATH" }
  ```

  De esta forma, tu código puede leer `os.environ["FACEBOOK_CREDENTIALS_PATH"]` y abrir el archivo.

---

## API de maws

### `build_lambda_handler(bot_type: str) -> Callable`

Devuelve un **handler de Lambda** ya configurado para el bot indicado.

* Inicializa perezosamente `AgentSystemManager`.
* Instancia el bot MAS (`WhatsappBot` / `TelegramBot`) con `verbose` desde `VERBOSE`.
* GET (WhatsApp): llama `handle_webhook_verification` del bot.
* POST: **auto-invoca** a la misma Lambda con payload “de trabajo” (modo background).
* En el “segundo salto”:

  * Toma **lock** por `user_id` (DynamoDB) si `LOCKS_TABLE_NAME` está definido; si el lock existe, **ignora** el update.
  * **Importa** historial desde S3 si existe.
  * Ejecuta `process_webhook_update`.
  * **Exporta** historial a S3.
  * **Libera** lock (o TTL lo limpiará).

No necesitás tocar nada en tu bot/clase base **MAS**.

---

## IAM / CloudFormation (ejemplo SAM)

Agregá algo similar a esto en tu `template.yaml` (resumen):

```yaml
Resources:
  ProcessingLocksTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${AWS::StackName}-locks"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions: [{ AttributeName: user_id, AttributeType: S }]
      KeySchema: [{ AttributeName: user_id, KeyType: HASH }]
      TimeToLiveSpecification: { AttributeName: expiresAt, Enabled: true }

  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_function.lambda_handler
      Runtime: python3.9
      CodeUri: .
      Policies:
        - AWSLambdaBasicExecutionRole
        - AWSXRayDaemonWriteAccess
        - Statement:
            - Effect: Allow
              Action: [ "ssm:GetParameter" ]
              Resource: !Sub arn:aws:ssm:${AWS::Region}:${AWS::AccountId}:parameter/my/param/name
        - Statement:
            - Effect: Allow
              Action: [ "s3:ListBucket" ]
              Resource: "arn:aws:s3:::<BUCKET_NAME>"
            - Effect: Allow
              Action: [ "s3:GetObject", "s3:PutObject" ]
              Resource: "arn:aws:s3:::<BUCKET_NAME>/*"
        - Statement:
            - Effect: Allow
              Action: [ "lambda:InvokeFunction" ]
              Resource: "*"
        - Statement:
            - Effect: Allow
              Action: [ "dynamodb:PutItem", "dynamodb:DeleteItem" ]
              Resource: !GetAtt ProcessingLocksTable.Arn
      Environment:
        Variables:
          BOT_TYPE: "whatsapp"
          VERBOSE: "false"
          BUCKET_NAME: "<BUCKET_NAME>"
          ENV_PARAMETER_NAME: "/my/app/prod/env"
          SYNC_TOKENS_S3: "true"
          TOKENS_S3_PREFIX: "secrets"
          SPECIAL_TOKEN_FILES_JSON: '[]'
          TOKEN_ENV_MAP_JSON: '{}'
          LOCKS_TABLE_NAME: !Ref ProcessingLocksTable
          LOCK_TTL_SECONDS: "180"
      Events:
        WebhookGET:
          Type: Api
          Properties: { Path: "/webhook", Method: get }
        WebhookPOST:
          Type: Api
          Properties: { Path: "/webhook", Method: post }
```

> Cambiá GET/POST según **WhatsApp** (GET+POST) o **Telegram** (POST-only).

---

## Patrón de auto-invocación

Para evitar timeouts de API Gateway, maws:

1. Responde **rápido** al POST (200 OK).
2. Se **auto-invoca** con el payload real para procesar en background.
3. Usa DynamoDB TTL como cerrojo por usuario (`user_id`) para evitar duplicados si llegan mensajes muy seguidos.

Si no configurás `LOCKS_TABLE_NAME`, el locking queda **deshabilitado** (comportamiento “best-effort”).

---

## Carga de entorno desde SSM

Si definís `ENV_PARAMETER_NAME`, maws leerá ese parámetro **SecureString** (estilo `.env`) y exportará **todas** sus claves al `os.environ` **sin** pisar variables ya presentes.

Formato admitido:

```
# comentario
KEY=VALUE
OTRA=VAL
```

---

## Historias de usuario / archivos

* Historias por usuario se guardan como SQLite en S3 bajo `history/<chat_id>.sqlite`.
* maws importa/expota automáticamente alrededor del procesamiento.
* Para **tokens** definidos en `SPECIAL_TOKEN_FILES_JSON`, sincroniza S3 → `/tmp` (o paquete → `/tmp` si no hay en S3) y opcionalmente expone rutas con `TOKEN_ENV_MAP_JSON`.

---

## Logging

* `VERBOSE=true` activa trazas detalladas en bots/MAS.
* CloudWatch Logs centraliza los logs de Lambda (costo bajo; recordá rotar).

---

## Requisitos

* Python 3.9+ (compatible con runtime `python3.9` de Lambda).
* AWS: S3, API Gateway, (opcional) DynamoDB, (opcional) SSM Parameter Store.
* Librerías: `mas`, `boto3`, `requests` (más las transitivas).

---

## Ejemplo mínimo end-to-end

**lambda\_function.py**

```python
import os
from maws import build_lambda_handler

lambda_handler = build_lambda_handler(os.environ.get("BOT_TYPE", "whatsapp"))
```

**requirements.txt**

```txt
boto3
requests
git+https://github.com/mneuronico/multi-agent-system-library.git#subdirectory=maws
git+https://github.com/mneuronico/multi-agent-system-library.git
```

**template.yaml**: ver sección IAM / CloudFormation.

---

## Preguntas frecuentes

**¿Puedo usar maws sin DynamoDB?**
Sí. Omití `LOCKS_TABLE_NAME` y se desactiva el locking.

**¿Qué pasa si no tengo tokens en S3?**
maws intentará copiarlos desde el **paquete** (raíz del zip) a `/tmp`. Si no existen, loguea aviso y sigue.

**¿Necesito MAS para usar maws?**
Sí. maws hace de “pegamento” entre AWS y **MAS**.

---

## Versionado y licencias

* Versionado en `maws/__init__.py` (ej. `__version__ = "0.1.0"`).
* Ver **LICENSE** en el repositorio principal.