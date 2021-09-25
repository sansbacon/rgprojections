"""
rg.py    

Library to scrape Rotogrinders projections

NOTE: Subscription is required
You must have firefox installed and have logged in to rotogrinders in that browser + profile
This library does not access to anything 'extra' that user didn't buy
It automates projections download for an authorized user (via browser_cookie3)

"""
import json
import re
from typing import Any, Dict, List, Union

import browser_cookie3
import numpy as np
import pandas as pd
import requests

from nflnames.players import *
from nflnames.teams import *
from nflprojections import ProjectionSource


class Scraper:
    """Scrape RG projections
    
    
    Example:
        params = {
          'site': 'draftkings', 
          'slate': '53019', 
          'projections': 'grid', 
          'date': '2021-09-12',
          'post': '2009661'    
        }

        s = Scraper()
        url = self.construct_url(params)
        content = self.get(url)

    """

    def __init__(self):
        """Creates Scraper instance"""
        self._s = requests.Session()
        self._s.cookies.update(browser_cookie3.firefox())

    def construct_url(self, params: dict) -> str:
        """Constructs URL given params
        
        Args:
            params (Dict[str, str]): the url parameters

        Returns:
            str

        """
        url = 'https://rotogrinders.com/lineuphq/nfl?site={site}&slate={slate}&projections=grid&date={date}&post={post}'
        return url.format(**params)


    def get(self, url: str, headers: dict = None, params: dict = None, return_object: bool = False) -> Union[str, Response]:
        """Gets HTML response"""
        r = self._s.get(url, headers=headers, params=params)
        if return_object:
            return r
        return r.text

    def lineuphq_projections(self, params: dict) -> str:
        """Gets projections given aparms"""
        return self.get(self.construct_url(params))


class Parser:
    """Parse projections html
    
    Examples:
        from pathlib import Path
        from nflprojections.rg import Parser

        pth = Path('html_file.html')
        html = pth.read_text()
        p = Parser()
        proj = p.lineuphq_projections(html)

    """
    PROJ_PATT = re.compile(r'selectedExpertPackage: ({.*?],"title":.*?"product_id".*?}),\s+serviceURL') 

    def lineuphq_projections(self, html: str, patt: re.Pattern = PROJ_PATT) -> List[Dict[str, Any]]:
        """Parses projections HTML"""
        match = re.search(patt, html, re.MULTILINE | re.DOTALL)
        return json.loads(match.group(1))['data']


class RotogrindersProjection(ProjectionSource):
    """Collects/parses projections from rotogrinders.com
       NOTE: subscription required, this is post-processing after valid download
    """

    COLUMN_MAPPING = {
        'player': 'plyr',
        'fpts': 'proj',
        'team': 'team',
        'opp': 'opp',
        'ceil': 'ceil', 
        'floor': 'floor',
        'playerid': 'source_player_id',
        'salary': 'salary'
    }

    def __init__(self, **kwargs):
        """Creates object"""
        kwargs['column_mapping'] = self.COLUMN_MAPPING
        kwargs['projections_name'] = 'rg-giq'
        super().__init__(**kwargs)

    def load_raw(self, html: str) -> pd.DataFrame:
        """Loads raw data from html
        
        Args:
            html (str): the page HTML

        Returns:
            pd.DataFrame

        """
        p = Parser()
        df = p.lineuphq_projections(html)
        df.columns = df.columns.str.lower()
        return df

    def process_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes raw dataframe
        
        Args:
            df (pd.DataFrame): the raw dataframe

        Returns:
            pd.DataFrame

        """
        df.columns = self.remap_columns(df.columns)

        # get rid of incomplete records
        df = df.dropna(thresh=7) 

        # convert types
        df['proj'] = pd.to_numeric(df['proj'], errors='coerce')
        df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
        df['ceil'] = pd.to_numeric(df['ceil'], errors='coerce')
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce', downcast='integer') 

        # fix missing values
        df.loc[:, 'floor'] = np.where(df['floor'] >= 0, df['floor'], (df['fpts'] - 1) / 4)
        df.loc[:, 'ceil'] = np.where(df['ceil'] >= 0, df['ceil'], (df['fpts'] * 1.5))

        return df

    def standardize(self, df):
        """Standardizes names/teams/positions
        
        Args:
            df (pd.DataFrame): the projections dataframe

        Returns
            pd.DataFrame

        """
        # standardize team and opp
        df = df.assign(team=self.standardize_teams(df.team), opp=self.standardize_teams(df.opp))

        # standardize positions
        df = df.assign(pos=self.standardize_positions(df.pos))

        # standardize player names
        return df.assign(plyr=lambda x: self.standardize_players(x))


if __name__ == '__main__':
    pass
