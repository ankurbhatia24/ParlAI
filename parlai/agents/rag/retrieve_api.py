#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
APIs for retrieving a list of "Contents" using an "Search Query".

The term "Search Query" here refers to any abstract form of input string. The definition
of "Contents" is also loose and depends on the API.
"""

from abc import ABC, abstractmethod
import requests
from typing import Any, Dict, List

from parlai.core.opt import Opt
from parlai.utils import logging

from googleapiclient.discovery import build
from urllib.request import urlopen
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re

CONTENT = 'content'
DEFAULT_NUM_TO_RETRIEVE = 5


class RetrieverAPI(ABC):
    """
    Provides the common interfaces for retrievers.

    Every retriever in this modules must implement the `retrieve` method.
    """

    def __init__(self, opt: Opt):
        self.skip_query_token = opt['skip_retrieval_token']

    @abstractmethod
    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        """
        Implements the underlying retrieval mechanism.
        """

    def create_content_dict(self, content: list, **kwargs) -> Dict:
        resp_content = {CONTENT: content}
        resp_content.update(**kwargs)
        return resp_content


class SearchEngineRetrieverMock(RetrieverAPI):
    """
    For unit tests and debugging (does not need a running server).
    """

    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        all_docs = []
        for query in queries:
            if query == self.skip_query_token:
                docs = None
            else:
                docs = []
                for idx in range(num_ret):
                    doc = self.create_content_dict(
                        f'content {idx} for query "{query}"',
                        url=f'url_{idx}',
                        title=f'title_{idx}',
                    )
                    docs.append(doc)
            all_docs.append(docs)
        return all_docs


class SearchEngineRetriever(RetrieverAPI):
    """
    Queries a server (eg, search engine) for a set of documents.

    This module relies on a running HTTP server. For each retrieval it sends the query
    to this server and receieves a JSON; it parses the JSON to create the the response.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt=opt)
        self.server_address = self._validate_server(opt.get('search_server'))

    def preprocess(self, text):
    
        text = text.encode("utf-8", errors='ignore').decode("utf-8")
        text = re.sub("https?:.*(?=\s)",'',text)
        text = re.sub("[’‘\"]","'",text)
        text = re.sub("[^\x00-\x7f]+",'',text)
        text = re.sub('[#&\\*+/<>@[\]^`{|}~ \t\n\r]',' ',text)
        text = re.sub('\(.*?\)','',text)
        text = re.sub('\=\=.*?\=\=','',text)
        text = re.sub(' , ',',',text)
        text = re.sub(' \.','.',text)
        text = re.sub("  +",' ',text)
        return text.strip()

    def google_scrape(self, url):
        try:
            thepage = urlopen(url).read().decode("utf-8")
            soup = BeautifulSoup(thepage, "html.parser")
            
            for script in soup(["script", "style"]):
                script.extract()    # rip it out
            
            text = soup.body.get_text()
            content = self.preprocess(text)
            content = "\n".join(sent_tokenize(content))    

            return content
        except:
            return None

    def google_search(self, search_term, api_key, cse_id, **kwargs):
        
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        
        return res['items']

    def _query_search_server(self, query_term, n):
        
        server = self.server_address
        req = {'q': query_term, 'n': n}
        logging.debug(f'sending search request to {server}')
        server_response = requests.post(server, data=req)
        resp_status = server_response.status_code
        #if resp_status == 200:
        #   return server_response.json().get('response', None)
        responses = []
        faulty = 0
        my_api_key = "AIzaSyCHwRw5chfOm0hvDb-LjVZM6BCquwcrpqM"
        my_cse_id = "c3abb6f8f234828ac" 
        results = self.google_search(query_term, my_api_key, my_cse_id, num=n)
        print('Search: ', query_term)
        for result in results:
            url, title = result['link'], result['title']
            text = self.preprocess(result['snippet']) if 'snippet' in result.keys() else None #
            #text = self.google_scrape(url)
            if text:
                dic = {
                    'url': url,
                    'title': title,
                    'content': text
                }
                responses.append(dic)
            else:
                faulty+=1
        print('Number of faulty links: ', faulty)
        if responses:
            return responses
        else:
            logging.error(
                f'Failed to retrieve data from server! Search server returned status {resp_status}'
            )

            return None


    def _validate_server(self, address):
        if not address:
            raise ValueError('Must provide a valid server for search')
        if address.startswith('http://') or address.startswith('https://'):
            return address
        PROTOCOL = 'http://'
        logging.warning(f'No portocol provided, using "{PROTOCOL}"')
        return f'{PROTOCOL}{address}'

    def _retrieve_single(self, search_query: str, num_ret: int):
        if search_query == self.skip_query_token:
            return None

        retrieved_docs = []
        search_server_resp = self._query_search_server(search_query, num_ret)
        if not search_server_resp:
            logging.warning(
                f'Server search did not produce any results for "{search_query}" query.'
                ' returning an empty set of results for this query.'
            )
            return retrieved_docs

        for rd in search_server_resp:
            url = rd.get('url', '')
            title = rd.get('title', '')
            sentences = [s.strip() for s in rd[CONTENT].split('\n') if s and s.strip()]
            retrieved_docs.append(
                self.create_content_dict(url=url, title=title, content=sentences)
            )
        return retrieved_docs

    def retrieve(
        self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE
    ) -> List[Dict[str, Any]]:
        # TODO: update the server (and then this) for batch responses.
        return [self._retrieve_single(q, num_ret) for q in queries]
