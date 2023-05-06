# py -m pip install ~/SentimentArcsPackage (or whatever the path is to the clone of the SentimentArcsPackage repo)
# py
# >>> import imppkg.simplifiedSA as SA

## every function that may raise an exception should be within a try except else block, like this:
#     try:
#         f = open(arg, 'r')
#     except OSError as error:
#         print('cannot open', arg)
#         print(f"Unexpected {error=}")
#     else:
#         print(arg, 'has', len(f.readlines()), 'lines')
#         f.close() 

# except Exception as error:
#     print(f"Unexpected {error=}, {type(error)=}") # print or log the exception
#     raise # raise it for the user to be able to catch (& see the standard traceback) as well