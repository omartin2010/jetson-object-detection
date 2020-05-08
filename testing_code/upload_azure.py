# from azure.storage.blob.aio import BlobServiceClient
# from azure.storage.blob.aio import ContainerClient
# from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
# # from azure.storage.blob.models import StorageErrorException
# from azure.storage.blob._generated.models._models_py3 import StorageErrorException
# from azure.identity.aio import EnvironmentCredential, ClientSecretCredential
# import os
# import datetime
# import asyncio
# import traceback


# blob_service_endpoint = "https://olibot.blob.core.windows.net/"
# container_name='videos'
# os.environ['AZURE_TENANT_ID'] = "7b2ae4eb-01ed-4d9b-9a3c-e4a510f51c98"
# tenant_id = os.environ['AZURE_TENANT_ID']
# os.environ['AZURE_CLIENT_ID'] = "674d46d3-ba30-4316-80e5-07126f6ca7ac"
# client_id = os.environ['AZURE_CLIENT_ID']
# os.environ['AZURE_CLIENT_SECRET'] = "..."
# client_secret = os.environ['AZURE_CLIENT_SECRET']
# creds = EnvironmentCredential()


# # Create container if doesn't exist already
# metadata = {
#     'create_time': datetime.datetime.now().strftime(
#         "%Y-%m-%d %H:%M:%S.%f"),
#     'type': 'video_file_container'
# }
# # blob_service_client = BlobServiceClient.from_connection_string(connection_string)
# blob_service_client = BlobServiceClient(account_url = blob_service_endpoint,credential=creds)
# name = 'video2'
# async with blob_service_client:
#     # Instantiate a new ContainerClient
#     container_client = blob_service_client.get_container_client(name)

#     try:
#         # Create new container in the service
#         await container_client.create_container(metadata=metadata)

#     except StorageErrorException:
#         print(f'container {name} already exists... storage exception')

#     except ResourceExistsError:
#         print(f'Container {name} already exists... resource exception')

#         # List containers in the storage account
#     my_containers = []
#     async for container in blob_service_client.list_containers():
#         my_containers.append(container)
#     # Delete the container
#     print('*'*10 + ' Container list ' + '*'*10)
#     for idx, container in enumerate(my_containers):
#         print(f'Container {idx} = {container}')
#     # await container_client.delete_container()

# target_filename = 'bob'
# blob_client = container_client.get_blob_client(target_filename)

# type(blob_client)


# blob_client.upload_blob()

# # testing asyncio
# print('go..')
# await asyncio.sleep(1)
# print('done')


# filename='bob.avi'
# os.path.join(datetime.datetime.now().strftime("%Y/%m/%d/%H"), filename)

# assert bobo, f'bobo is {bobo} '\
#              f'none!!'