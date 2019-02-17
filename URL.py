'''
Write a program that, given a URL, parses out and displays its constituent components. For the purpose of this challenge, a url is in the form <protocol>://<domain>/path?<query_string>. Print out the protocol, domain and query_string, separated by commas.  
'''

from urllib.parse import urlparse
# from urlparse import urlparse  # Python 2
parsed_uri = urlparse('http://stackoverflow.com/questions/1234567/blah-blah-blah-blah' )
# result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
# print(result)
print(parsed_uri.scheme, parsed_uri.netloc, parsed_uri.path, sep=", ")
