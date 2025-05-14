🔧 Assign Cosmos DB Built-in Data Contributor to a Web App (System-Assigned Identity)
✅ Prerequisites
Cosmos DB account (e.g. cosmostest1234567890)
Web App with system-assigned identity enabled (e.g. webapp)
Azure CLI installed and authenticated
Step 1 – Confirm Role Exists
Check if the built-in role is available:

az cosmosdb sql role definition list \
  --account-name <COSMOS_DB_ACCOUNT_NAME> \
  --resource-group <COSMOS_RESOURCE_GROUP>
Look for:

"roleName": "Cosmos DB Built-in Data Contributor"
"name": "00000000-0000-0000-0000-000000000002"
Step 2 – Get Web App Identity
Fetch the principal ID of the Web App’s system-assigned identity:

az resource show \
  --name <WEB_APP_NAME> \
  --resource-group <WEB_APP_RESOURCE_GROUP> \
  --resource-type "Microsoft.Web/sites" \
  --query identity.principalId \
  -o tsv
Save the output (e.g. abc12345-6789-def0-1111-222233334444)

Step 3 – Assign the Role
Replace all placeholders and run:

az cosmosdb sql role assignment create \
  --account-name <COSMOS_DB_ACCOUNT_NAME> \
  --resource-group <COSMOS_RESOURCE_GROUP> \
  --role-definition-id "00000000-0000-0000-0000-000000000002" \
  --principal-id <PRINCIPAL_ID_FROM_STEP_2> \
  --scope "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<COSMOS_RESOURCE_GROUP>/providers/Microsoft.DocumentDB/databaseAccounts/<COSMOS_DB_ACCOUNT_NAME>"
🧪 Example Use in Code (Python)
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

endpoint = "https://<COSMOS_DB_ACCOUNT_NAME>.documents.azure.com"
credential = DefaultAzureCredential()
client = CosmosClient(endpoint, credential)

# Access a DB/container
database = client.get_database_client("<DATABASE_NAME>")
container = database.get_container_client("<CONTAINER_NAME>")
container.upsert_item({ "id": "test-doc", "message": "hello cosmos" })
