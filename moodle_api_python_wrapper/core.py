# Standard library
import copy
import itertools
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third-party
import networkx as nx
import numpy as np
import pytz
import requests
import sklearn
import xmltodict
from django.shortcuts import get_object_or_404
from django.urls import reverse
from pandas import DataFrame, DateOffset, concat, read_sql, to_datetime
from pytube import YouTube
from requests import get, post
from sklearn.mixture import GaussianMixture
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

# Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio



mcq_templatemoduleid=36
sw_templatemoduleid=39
demo_courseid=56
demo_sectionid=379


def _make_moodle_engine(moodle_db) -> Optional[object]:
    """Create engine lazily; no work at import time."""
    if 'moodle_db' not in globals():
        return None
    cfg = moodle_db.get('moodle') if isinstance(moodle_db, dict) else None
    if not cfg:
        return None
    url = f"mysql+mysqlconnector://{cfg['USER']}:{cfg['PASSWORD']}@{cfg['HOST']}/{cfg['NAME']}"
    engine = create_engine(url, pool_pre_ping=True)
    # no test connect() here; let first usage trigger connect
    return engine



class MgMoodle:
    def __init__(self,mWAParams,moodle_db):
        self.mWAP=mWAParams
        self.engine=_make_moodle_engine(moodle_db) 
        self.hvp_local_methods={}
        self.hvp_local_methods['update_hvp_MCQ_questions']=self.update_hvp_MCQ_questions
        self.hvp_local_methods['update_hvp_SW']=self.update_hvp_SW
        self.hvp_local_methods['update_hvp_videointeractions_MCQ']=self.update_hvp_videointeractions_MCQ
        self.hvp_local_methods['update_hvp_fill_in_the_blanks']=self.update_hvp_fill_in_the_blanks
        self.hvp_local_methods['get_hvp_video_interactions']=self.get_hvp_video_interactions
        self.hvp_local_methods['get_hvp_MCQ_questions']=self.get_hvp_MCQ_questions
        self.hvp_local_methods['get_hvp_SW_questions']=self.get_hvp_SW_questions
        self.hvp_local_methods['update_hvp_drag_the_words']=self.update_hvp_drag_the_words


#################################
#General methods
###############################

    def wrapper_fn(self,func): 
        
        def inner1(*x,**a): 
            c=func(*x,**a) 
            return c
                
        return inner1 

    def get_methods_info(self, user_id=None):
        method_dicts=[]
        # Introspect class methods
        methods = inspect.getmembers(self, inspect.ismethod)

        for name, method in methods:
            # Skip private methods
            if name.startswith('_'):
                continue

            # Get method signature
            signature = inspect.signature(method)

            # Get docstring
            docstring = method.__doc__
            formatted_docstring = docstring.strip() if docstring else "No description available"

            # Append method documentation
            method_dicts+= [{"method": name, "signature":str(signature), "description": formatted_docstring}]
        status='success'
        return {'status':status,'response':method_dicts}

    def get_sql_request(self,sqlQ, user_id=None):
        status='error'
        sql_response=[]
        try:
            self.engine.dispose()
            con=self.engine.connect()
            sql_response=read_sql(sql_text(sqlQ),con).to_dict(orient='records')
            status='success'
        except:
            pass

        con.close()
        data_json=json.dumps({"records":sql_response})
        message=f'SQL Query: {sqlQ}'
        meta_data=message
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response}
        
    def get_keys(self,dictionary, user_id=None):
        result = []
        for key, value in dictionary.items():
            if type(value) is dict:
                new_keys = self.get_keys(value)
                result.append(key)
                for innerkey in new_keys:
                    result.append(f'{key}/{innerkey}')
            else:
                result.append(key)
        return result

    def get_fig_menus(self,numberofmenus,datadicts, user_id=None):
        menus=()
        status='error'
        if len(datadicts)!=0:
            options=[{'label':ky,'value':ky} for ky in datadicts[0]]
            for nn in range(numberofmenus):
                menus+=options,
            status='success'
        else:
            pass
        return {'status':status, 'response':menus}

    def hvp_function_call(self,functionName,*fnVaraibles, user_id=None,**functionParameters):
        '''functionParameters is a dictionary of the parameters to be passed to the function'''   
        function_to_be_called = self.wrapper_fn(functionName)  
        return function_to_be_called(*fnVaraibles,**functionParameters)

#######################################
#Moodle webservice call
#######################################

    def _rest_api_parameters(self,in_args, prefix='', out_dict=None):
        """Transform dictionary/array structure to a flat dictionary, with key names
        defining the structure.
        Example usage:
        >>> _rest_api_parameters({'courses':[{'id':1,'name': 'course1'}]})
        {'courses[0][id]':1,
        'courses[0][name]':'course1'}
        """
        if out_dict==None:
            out_dict = {}
        if not type(in_args) in (list,dict):
            out_dict[prefix] = in_args
            return out_dict
        if prefix == '':
            prefix = prefix + '{0}'
        else:
            prefix = prefix + '[{0}]'
        if type(in_args)==list:
            for idx, item in enumerate(in_args):
                self._rest_api_parameters(item, prefix.format(idx), out_dict)
        elif type(in_args)==dict:
            for key, item in in_args.items():
                self._rest_api_parameters(item, prefix.format(key), out_dict)
        return out_dict

    def _call(self,accessParams, fname, **kwargs):
        """Calls moodle API function with function name fname and keyword arguments.
        Example:
        >>> call_mdl_function('core_course_update_courses',
                            courses = [{'id': 1, 'fullname': 'My favorite course'}])
        """
        parameters = self._rest_api_parameters(kwargs)
        parameters.update({"wstoken": accessParams['KEY'], 'moodlewsrestformat': 'json', "wsfunction": fname})
        response = post(accessParams['URL']+accessParams['ENDPOINT'], parameters,verify=False)
        response = response.json()
        if type(response) == dict and response.get('exception'):
            raise SystemError("Error calling Moodle API\n", response)
        return response

    def call_moodle_webservice(self,wsfunction,parameters, user_id=None):
        '''
        {
            "arguments":{"wsfunction":<the webservice function name>,"parameters":<kwargs>},
            "response": "Webservice response"
        }
        '''
        response=[]
        status="error"
        try:
            response=self._call(self.mWAP,wsfunction,**parameters)
            status='success'
        except Exception as e:
            status = f"Error: {str(e)}"

        return {'status':status,'response':response}


#######################################
#Moodle quiz creation
#######################################

    def create_moodle_quiz(self,questionsdict, user_id=None):
        '''
        
        '''
        answerdict={'@fraction': '0',
                    '@format': 'html',
                    'text': '<p dir="ltr" style="text-align: left;">Choice<br></p>',
                    'feedback': {'@format': 'html', 'text': 'none'}}
        questionict={'@type': 'multichoice',
                    'name': {'text': 'Quation Name1'},
                    'questiontext': {'@format': 'html','text': '<p dir="ltr" style="text-align: left;"></p><p>Text 1<p></p>'},
                    'generalfeedback': {'@format': 'html', 'text': 'none'},
                    'defaultgrade': '1',
                    'penalty': '0',
                    'hidden': '0',
                    'idnumber': 'none',
                    'single': 'true',
                    'shuffleanswers': 'true',
                    'answernumbering': 'none',
                    'showstandardinstruction': '0',
                    'correctfeedback': {'@format': 'html', 'text': 'Your answer is correct.'},
                    'partiallycorrectfeedback': {'@format': 'html',
                    'text': 'Your answer is partially correct.'},
                    'incorrectfeedback': {'@format': 'html', 'text': 'Your answer is incorrect.'},
                    'shownumcorrect': 'none',
                    'answer': [answerdict]}
        xml_dict={'quiz':{'question':[questionict]}}
        question=[]
        for dct in questionsdict:
            qdct=copy.deepcopy(questionict)
            qdct['name']['text']=dct['name']
            dct.pop('name')
            qdct['defaultgrade']=dct['grade']
            dct.pop('grade')
            qdct['questiontext']['text']= '<p dir="ltr" style="text-align: left;"></p><p>{}<p></p>'.format(dct['question'])
            dct.pop('question')
            answer=[]
            for ky in [*dct]:
                adct=copy.deepcopy(answerdict)
                if dct[ky] not in ['',None]:
                    if ky=='correct':
                        adct['@fraction']='100'
                        adct['text']='<p dir="ltr" style="text-align: left;">{}<br></p>'.format(dct[ky])
                    else:
                        adct['@fraction']='0'
                        adct['text']='<p dir="ltr" style="text-align: left;">{}<br></p>'.format(dct[ky])
                    answer+=[adct]
                    qdct['answer']=answer
            question+=[qdct]
        xml_dict={'quiz':{'question':question}}
        xml_data = xmltodict.unparse(xml_dict, pretty=True)
        status='success'
        message="Moodle XML quiz generated"
        meta_data=message
        response = {"meta_data": meta_data, "xml":xml_data, "message":message}
        return {"status": status, "response": response} 


############################################################
#Moodle methods
###########################################################

    def get_category_info(self, categoryids=None, user_id=None):
        """
        Retrieve category information from the `mdl_course_categories` table.

        This method queries the database for course category records. If specific category IDs
        are provided, it filters the query to include only those IDs. Otherwise, it retrieves all categories.
        The resulting data includes name, ID, parent category, ID number, and a cleaned version of the description
        (stripped of HTML tags).

        Parameters:
            categoryids (list[int], optional): A list of category IDs to filter the results. 
                                            If None, all categories are retrieved.

        Returns:
            dict: A dictionary containing:
                - 'status' (str): 'success' if the operation succeeded, or 'error' with message on failure.
                - 'response' (dict): 
                    - 'meta_data' (str): Informational message about the operation status.
                    - 'data' (str): JSON string of the category records.
                    - 'message' (str): Descriptive status or error message.

        Notes:
            - The method ensures that HTML tags are stripped from the `description` field.
            - The database connection is disposed and reestablished before querying.
            - Any exceptions are caught and returned in the response message.

        Example:
            >>> get_category_info([1, 2, 3])
            {
                'status': 'success',
                'response': {
                    'meta_data': 'Category Infomartion Retrieved',
                    'data': '{"records": [...]}',
                    'message': 'Category Infomartion Retrieved'
                }
            }
        """

        response=[]
        status='error'
        message=status
        self.engine.dispose()
        con=self.engine.connect()
        try:
            if categoryids:
                sqlQ="SELECT cc.name, cc.id, cc.parent, cc.idnumber, cc.description FROM mdl_course_categories cc WHERE cc.id IN {}".format(tuple(categoryids+[0]))
            else:
                sqlQ="SELECT mcc.name, mcc.id, mcc.parent, mcc.idnumber, mcc.description FROM mdl_course_categories mcc"
            df=read_sql(sql_text(sqlQ),con)
            df['description']=df['description'].str.replace(r'<[^<>]*>', '', regex=True)
            response=df.to_dict(orient='records')
            status='success'
            message="Category Infomartion Retrieved"
            data_id=str(uuid.uuid4())
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            data_id=None
            
        con.close()
        
        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def get_user_category_instances_by_role(self, userid, rolename, user_id=None):
        response=[]
        status='error'
        self.engine.dispose()
        con=self.engine.connect()
        try:
            sqlQ = "SELECT mcc.path, mcc.name AS label, mcc.id AS value FROM mdl_role_assignments ra INNER JOIN mdl_role r ON r.id=ra.roleid INNER JOIN mdl_context ctx ON ra.contextid=ctx.id INNER JOIN mdl_course_categories mcc ON mcc.id=ctx.instanceid WHERE ra.userid={} AND r.shortname='{}' AND ctx.contextlevel=40".format(userid,rolename)
            response=read_sql(sql_text(sqlQ),con).to_dict(orient='records')
            status='success'
        except:
            pass

        con.close()
        return {'status':status,'response':response}

    def get_user_category_list(self,userid, user_id=None):
        """
        Retrieves a list of course categories accessible to a user based on their role.

        This method determines whether a user is a system manager or category manager and 
        fetches the relevant course categories accordingly.

        Parameters:
            userid (int): The unique ID of the user whose category access is being retrieved.

        Returns:
            dict: A dictionary containing:
                - 'status' (str): 'success' if the operation succeeded, otherwise 'error'.
                - 'response' (dict):
                    - 'categorylist' (list): A list of accessible course categories for the user.
                    - 'rolename' (str): The role of the user — either 'systemmanager', 'categorymanager', or ''.

        Logic:
            1. Checks if the user has a system-wide 'manager' role by querying with contextlevel 10.
            2. If found, sets role to 'systemmanager' and fetches all categories.
            3. If not, checks for category-level 'manager' roles (contextlevel 40).
            4. If found, sets role to 'categorymanager' and fetches categories where the user is a manager.
            5. Returns the user's role and the corresponding list of categories.

        Notes:
            - Errors are silently passed (exception is caught without handling).
            - Relies on internal methods:
                - `get_user_context_instance_roles`
                - `get_category_list`
                - `get_user_context_instances`
                - `get_user_category_instances_by_role`

        Example:
            >>> get_user_category_list(42)
            {
                'status': 'success',
                'response': {
                    'categorylist': [...],
                    'rolename': 'categorymanager'
                }
            }
        """

        categorylist=[]
        role=''
        status='error'
        message=status
        try:
            data={'userid':userid,'contextlevel':10,'instanceid':0}
            systemcontexts=self.get_user_context_instance_roles(**data)['response']
            if 'manager' in [dct['rolename'] for dct in systemcontexts]:
                role='systemmanager'
                categorylist=self.get_category_list()['response']
            else:
                data={"userid":userid,"contextlevel":40}
                categoryinstances=self.get_user_context_instances(**data)['response']
                if 'manager' in [dct['rolename'] for dct in categoryinstances]:
                    role='categorymanager'
                data={'userid':userid,'rolename':'manager'}
                categorylist=self.get_user_category_instances_by_role(**data)['response']
            status='success'
            message=status
        except:
            pass
        return {'status':status,'response':{'categorylist':categorylist,'rolename':role}}
    
    def get_categories_course_info(self,categoryids=None, user_id=None):
        """
        Retrieves course information for one or more given course categories.

        This method queries the `mdl_course` table to fetch details about courses that belong
        to the provided category IDs. It includes course short name, full name, start and end dates,
        which are converted from UNIX timestamps to human-readable datetime format in the Asia/Colombo timezone.

        Parameters:
            categoryids (list[int], required): A list of category IDs to filter the courses. 
                                            If not provided, the function returns an error message.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if the operation was successful, otherwise an error message.
                - 'response' (dict):
                    - 'meta_data' (str): Informational message about the query result or error.
                    - 'data' (str): JSON string containing a list of course records.
                    - 'message' (str): Same as meta_data, used for redundancy in frontends.

        Course Fields Returned:
            - shortname (str): The short name of the course.
            - fullname (str): The full name of the course.
            - coursestartdate (str): The start date in '%Y-%m-%d %X' format (Asia/Colombo timezone).
            - courseenddate (str): The end date in '%Y-%m-%d %X' format (Asia/Colombo timezone).

        Notes:
            - Ensures HTML formatting is removed from the response (if applicable).
            - Uses pandas `to_datetime` to convert UNIX timestamps.
            - Uses `tuple(categoryids + [-1])` to safely handle single-element lists in SQL IN clause.

        Example:
            >>> get_categories_course_info([3, 5])
            {
                'status': 'success',
                'response': {
                    'meta_data': 'success',
                    'data': '{"records": [...]}',
                    'message': 'success'
                }
            }
        """
        response=[]
        status='error'
        message=status
        self.engine.dispose()
        con=self.engine.connect()
        if categoryids:
            try:
                sqlQ="SELECT mc.id, mc.shortname, mc.fullname, mc.startdate AS coursestartdate, mc.enddate AS courseenddate FROM mdl_course mc WHERE mc.category IN {}".format(tuple(categoryids+[-1]))
                df=read_sql(sql_text(sqlQ),con)
                df['coursestartdate'] = to_datetime(df['coursestartdate'],unit='s',utc=True).dt.tz_convert('Asia/Colombo').dt.strftime('%Y-%m-%d %X')
                df['courseenddate'] = to_datetime(df['courseenddate'],unit='s',utc=True).dt.tz_convert('Asia/Colombo').dt.strftime('%Y-%m-%d %X')
                response=df.to_dict(orient='records')
                status='success'
                message=status
                data_id=str(uuid.uuid4())
            except Exception as e:
                status=f'Error: {str(e)}'
                message=status
                data_id=None
        else:
            status=f'No categoryids provided.'
            message=status
            data_id=None


        con.close()
        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def get_users_list(self, user_id=None):
        """
        Retrieves a list of all users from the Moodle system.

        This method calls the Moodle Web API (`core_user_get_users`) with empty search criteria 
        to fetch all available users. Each user is returned with their email as a label and ID as value,
        suitable for use in dropdowns or selectors.

        Returns:
            dict: A dictionary containing:
                - 'status' (str): 'success' if users were fetched successfully, otherwise 'error' with a message.
                - 'response' (dict):
                    - 'meta_data' (str): Status message or error message.
                    - 'data' (str): JSON-encoded string with a list of user records.
                    - 'message' (str): Duplicate of `meta_data`, for frontend convenience.

        User Record Format:
            - label (str): User's email address.
            - value (int): User's ID.

        Notes:
            - Uses empty criteria `{"key": "", "value": ""}` to fetch all users. This may return a large dataset.
            - Error handling captures and returns any exception as part of the response.

        Example:
            >>> get_users_list()
            {
                'status': 'success',
                'response': {
                    'meta_data': 'success',
                    'data': '{"records":[{"label":"john@example.com","value":3}, ...]}',
                    'message': 'success'
                }
            }

        Raises:
            Exception: If the API call fails or returns malformed data, the error is caught and included in the output.
        """
        response=[]
        status='error'
        message=status
        try:
            criteria=[{"key":"","value":""}]
            users=self._call(self.mWAP,'core_user_get_users',criteria=criteria)['users']
            if users:
                response=[{'label':usr['email'],'value':usr['id']} for usr in users]
            status='success'
            message=status
            data_id=str(uuid.uuid4())
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            data_id=None

        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response}       

    def get_user_courses(self,userid, hidden=0, user_id=None):
        """
        Retrieves the list of courses a user is enrolled in, with optional filtering for hidden courses.

        This method uses the Moodle Web API to:
        1. Check if the user is suspended.
        2. If not suspended, fetch all courses the user is enrolled in.
        3. Filter courses based on the 'hidden' status.

        Parameters:
            userid (int): The ID of the user for whom the course list is to be retrieved.
            hidden (int, optional): Filter flag for course visibility.
                                    - 0 (default): Only visible (non-hidden) courses are returned.
                                    - 1: Only hidden courses are returned.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 
                    - 'success' if courses were fetched,
                    - 'user suspended' if the user is marked suspended in Moodle,
                    - or 'error' with exception message if an error occurred.
                - 'response' (dict):
                    - 'meta_data' (str): Informational status or error message.
                    - 'data' (str): JSON-encoded string of course records in `{ value, label }` format.
                    - 'message' (str): Same as `meta_data` for frontend usage.

        Course Record Format:
            - value (int): Course ID.
            - label (str): Course short name.

        Example:
            >>> get_user_courses(21)
            {
                'status': 'success',
                'response': {
                    'meta_data': 'success',
                    'data': '{"records":[{"value":12,"label":"Math101"},{"value":15,"label":"Bio202"}]}',
                    'message': 'success'
                }
            }

        Raises:
            Exception: Any errors during API calls or data processing are caught and returned in the status.
        """
        response=[]
        status='error'
        message=status
        try:
            usersDict=self._call(self.mWAP,'core_user_get_users',criteria=[{'key':'id','value':userid}])
            if not usersDict['users'][0]['suspended']:
                response=[{'value':crs['id'],'label':crs['shortname']} for crs in self._call(self.mWAP,'core_enrol_get_users_courses', userid=userid) if crs['hidden']==hidden]
                status='success'
            else:
                response=[]
                status='user suspended'
            message=status
            data_id=str(uuid.uuid4())
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            data_id=None

        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 
 
    def get_courses_users(self,courseids, user_id=None):
        """
        Retrieves enrolled user information for a list of course IDs.

        This method calls the Moodle Web API to fetch enrolled users for each course ID provided. 
        The response includes user details such as ID, first name, username, full name, and assigned roles.

        Parameters:
            courseids (list[int]): A list of Moodle course IDs for which to fetch enrolled user information.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if users were fetched, or 'error' with a message otherwise.
                - 'response' (dict):
                    - 'meta_data' (str): Informational status or error message.
                    - 'data' (str): JSON-encoded string of user records.
                    - 'message' (str): Same as meta_data, included for compatibility.

        Notes:
            - Calls the internal method `call()` with the Moodle Web API method `core_enrol_get_enrolled_users`.
            - Users are sorted by `username` before being added to the response.
            - Each user record includes: course ID, user ID, first name (mapped from sorted username), username, full name, and roles.
            - Deduplicates user entries using username-based matching during list comprehension.

        Example:
            >>> get_courses_users([12, 14])
            {
                'status': 'success',
                'response': {
                    'meta_data': {"message":message, "data_id":uuid data_id, "fields":columns}
                    'data': '{"records": [...]}',
                    'message': 'success'
                }
            }

        Raises:
            Exception: If API call fails or data processing encounters issues. Error message is captured in status.
        """

        response=[]
        columns=[]
        status="error"
        message=status
        try:
            for courseid in courseids:
                courseUsers=self._call(self.mWAP,'core_enrol_get_enrolled_users',courseid=courseid)
                if len(courseUsers)!=0:
                    labelNames=[dct['username'] for dct in courseUsers]
                    labelNames.sort()
                    response+=[{'courseid':courseid,'id':dct['id'],'firstname':lbl, 'username':dct['username'], 'fullname':dct['fullname'], 'roles':dct['roles']} for lbl in labelNames for dct in courseUsers  if dct['username']==lbl]
            status='success'
            columns=list(response[0].keys())
            message=f'Data with columns {columns} retrieved.'
            data_id=str(uuid.uuid4())
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            data_id=None

        meta_data={"message":message, "data_id":data_id, "fields":columns}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def get_moodle_system_roles(self, rolename, user_id=None):
        """
        Retrieves Moodle system role information by shortname.

        This method connects to the database and queries the `mdl_role` table
        to fetch role details for the given shortname (`rolename`). It returns
        structured metadata and JSON-formatted results.

        Parameters:
            rolename (str): The shortname of the Moodle role to search for.
                            Example: 'student', 'editingteacher', 'admin'

        Returns:
            dict: A dictionary containing the following keys:
                - 'status': str — "success" or error message.
                - 'response': dict with:
                    - 'meta_data': {
                        'fields': list of column names from the result,
                        'data_id': UUID string (if data was found),
                        'message': description of outcome
                    }
                    - 'data': JSON string representing query results, in format {"records": [...]}
                    - 'message': Summary of result or error details.

        Notes:
            - If no role matches the given shortname, the result will still be successful
            but contain an empty data list and appropriate message.
            - On database or query failure, the status will include the error message.
        """

        response=[]
        columns=[]
        data_id=None
        status='error'
        self.engine.dispose()
        con=self.engine.connect()        
        try:
            sqlQ="SELECT mr.shortname AS rolename, mr.id FROM mdl_role mr WHERE mr.shortname='{}'".format(rolename)
            response=read_sql(sql_text(sqlQ),con).to_dict(orient='records')   
            status='success'
            if isinstance(response,list) and response:
                columns=list(response[0].keys())
                message=f'The moodle system roles are: {response}'
                data_id=str(uuid.uuid4())
            else:
                message='No data retrieved.'
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status

        con.close()

        meta_data={"fields":columns, "data_id":data_id, "message":message}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def get_instance_contextid(self,contextlevel,instanceid, user_id=None):
        response=None
        status='error'
        self.engine.dispose()
        con=self.engine.connect()
        try:
            sqlQ="SELECT id FROM mdl_context WHERE contextlevel = {} AND instanceid = {}".format(contextlevel,instanceid)
            response=read_sql(sql_text(sqlQ),con).to_dict(orient='records')[0]["id"]
            # sqlQ = text("SELECT id FROM mdl_context WHERE contextlevel = :contextlevel AND instanceid = :instanceid")
            # response = read_sql(sqlQ, con, params={"contextlevel": contextlevel, "instanceid": instanceid}) \
            #             .to_dict(orient='records')[0]["id"]


            status='success'
            if response:
                message=f'Context id of instanceid {instanceid} of contextlevel {contextlevel} is {response}.'
            else:
                message='No data retrieved.'            
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status

        con.close()

        meta_data={"instanceid":instanceid, "contextlevel":contextlevel, "contextid":response}
        data_json=json.dumps({"contextid":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def enroll_users_moodle_course(self, datadicts, user_id=None):
        """
        Enrolls users into Moodle courses using the manual enrollment plugin.

        This method sends a list of enrollment data dictionaries to the Moodle
        web service API `enrol_manual_enrol_users`. It then returns a structured
        response with metadata and a summary message.

        Each dictionary in `datadicts` must follow Moodle's expected structure for enrolments:
            {
                "roleid": int,       # ID of the role (e.g., student, teacher)
                "userid": int,       # ID of the user to be enrolled
                "courseid": int,     # ID of the course
                "timestart": int,    # (Optional) Unix timestamp for when enrolment starts
                "timeend": int,      # (Optional) Unix timestamp for when enrolment ends
                "suspend": int       # (Optional) 1 to suspend the user, 0 otherwise
            }

        Parameters:
            datadicts (List[Dict]): A list of enrollment dictionaries to be passed to Moodle's API.

        Returns:
            dict: A structured response in the following format:
                {
                    "status": str,  # "success" or an error message
                    "response": {
                        "meta_data": {
                            "fields": List[str],    # List of field names (if any returned)
                            "data_id": str or None, # UUID for this data session
                            "message": str          # Human-readable result or error message
                        },
                        "data": str,               # JSON-encoded string of response records
                        "message": str             # Same as meta_data["message"]
                    }
                }

        Notes:
            - If Moodle API returns an empty list, it's assumed the enrolments succeeded without return data.
            - Any exceptions during the API call are captured and returned in the response message.
            - `uuid.uuid4()` is used to tag the data session if data is returned.
        """


        response=[]
        columns=[]
        data_id=None
        status="error"
        try:
            response=self._call(self.mWAP,'enrol_manual_enrol_users',enrolments=datadicts)
            status='success'
            if isinstance(response,list) and response:
                columns=list(response[0].keys())
                message=f'Data with columns {columns} retrieved.'
                data_id=str(uuid.uuid4())
            else:
                message='Users enrolled.'
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            


        meta_data={"fields":columns, "data_id":data_id, "message":message}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def assign_context_roles(self, datadicts, user_id=None):
        """
        Assigns roles to users within specific Moodle contexts.

        This method uses the Moodle web service API `core_role_assign_roles` to assign
        roles such as teacher, student, or manager to users in a specified context
        (e.g., course, user, system).

        Each dictionary in `datadicts` must follow Moodle's required structure for role assignments:
            {
                "roleid": int,       # ID of the role to assign (e.g., student, teacher)
                "userid": int,       # ID of the user
                "contextid": int     # ID of the context (from mdl_context)
            }

        Parameters:
            datadicts (List[Dict]): A list of role assignment dictionaries to be passed to Moodle's API.

        Returns:
            dict: A structured response in the following format:
                {
                    "status": str,  # "success" or an error message
                    "response": {
                        "meta_data": {
                            "fields": List[str],     # List of keys returned in API response (if any)
                            "data_id": str or None,  # UUID representing the data transaction
                            "message": str           # Result summary or error message
                        },
                        "data": str,                # JSON string of the response payload
                        "message": str              # Same as meta_data["message"]
                    }
                }

        Notes:
            - If the API response is a non-empty list, the method will extract column names
            and assign a unique `data_id`.
            - If the API response is empty or a string message, it is assumed successful and passed through.
            - Any exceptions during the call will be captured and returned in the response.
        """


        response=[]
        columns=[]
        data_id=None
        status="error"
        try:
            response=self._call(self.mWAP,'core_role_assign_roles',assignments=datadicts)
            status='success'
            if isinstance(response,list) and response:
                columns=list(response[0].keys())
                message=f'Data with columns {columns} retrieved.'
                data_id=str(uuid.uuid4())
            else:
                message=response #'No data retrieved.'
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            


        meta_data={"fields":columns, "data_id":data_id, "message":message}
        data_json=json.dumps({"records":response})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 

    def get_courses_sections_list(self, courseids, secnvisible=1, user_id=None):
        '''
        Webserice call: {"categoryid":<Category ID>,"visible":<1 or 0>}
        Response: {'status':status,'response':response}
        '''

        response=[]
        columns=[]
        records=[]
        status="error"
        message=status
        dropdownlist=[]
        secnInfo=[]
        errors=[]
        for courseid in courseids:
            try:
                records+=[{'courseid':courseid,'sectionid':secn['id'], 'sectionname':secn['name']} for secn in self._call(self.mWAP,'core_course_get_contents',courseid=courseid) if secn['visible']==secnvisible]             
            except Exception as e:
                errors.append({'error':str(e)})
            
        if len(records)!=0:
            status='success'
            columns=records[0].keys()
            message=f'Data with columns {columns} retrieved.'
            data_id=str(uuid.uuid4())
        else:
            status=f'No sections returned.'
            message=status
            data_id=None

        meta_data={"message":message, "data_id":data_id, "fields":columns, 'errors':errors}
        data_json=json.dumps({"records":records})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response} 
        
    def get_courses_modules_list(self, courseids, user_id=None):
        """
        Retrieves the list of modules for the given course IDs.

        For each course ID provided, this method fetches course content using the
        'core_course_get_contents' webservice call, extracts module details from each section,
        and returns a structured response with metadata, module records, and error logs.

        Args:
            courseids (list): A list of course IDs (integers) to fetch module information for.
            user_id (int, optional): The user ID requesting the data. Included for potential
                                    future use or permission filtering.

        Returns:
            dict: A dictionary with:
                - 'status' (str): "success" if modules are found, "error" otherwise.
                - 'response' (dict):
                    - 'meta_data' (dict): Contains:
                        - 'fields' (list): List of module field names.
                        - 'data_id' (str): Unique identifier for this data retrieval.
                        - 'message' (str): Informational message about the operation.
                    - 'data' (str): JSON string of the records, each representing a module.
                    - 'message' (str): Summary message of the result.
                    - 'errors' (list): List of error messages encountered during the process.

        Example:
            result = self.get_courses_modules_list([101, 202], user_id=5)
        """

        status="error"
        data_json='{}'
        meta_data={}
        columns=[]
        records=[]
        message=''
        data_id=''
        errors=[]
        course_content=None
        course_info=[]
        course_name=''
        modules_info=[]
        try:
            for courseid in courseids:
                try:
                    course_info=self._call(self.mWAP,'core_course_get_courses',options={'ids':[courseid]})
                    course_content=self._call(self.mWAP,'core_course_get_contents',courseid=courseid)
                    if isinstance(course_info,list) and course_info:
                        course_name=course_info[0].get('displayname','')
                except Exception as e:
                    errors.append(str(e))
                    course_content=None

                if course_content:
                    for secn in course_content:
                        if secn['modules']:
                            for mod in secn['modules']: 
                                modules_info.append({'courseid':courseid,'coursename':course_name,'sectionid':secn['id'], 'sectionname':secn['name'],'moduleid':mod['id'], 'modulename':mod['name'], 'modname':mod['modname'],'contextid':mod['contextid'],'instance':mod['instance'],'url':mod['url']})
                course_name=''
                course_info=[]

            if modules_info:
                status='success'
                data_id=str(uuid.uuid4())
                records=modules_info
                columns=modules_info[0].keys()
                message=f"Data returned with columns {', '.join(columns)}"
            else:
                message="No data returned."

        except Exception as e:
            status=f'Error: {str(e)}'
            message=status

        meta_data={"fields":columns, "data_id":data_id, "message":message}
        data_json=json.dumps({"records":records})
        response = {"meta_data": meta_data, "data":data_json, "message":message, "errors":errors}
        return {"status": status, "response": response}    

#####################################################
#Course/Section/Module Creation
####################################################

    def duplicate_course(self,courseid,categoryid,shortname,fullname, user_id=None):  
        '''
        Webserice call: {"courseid":<courseid>,"categoryid":<Category ID>,"shortname":<Course short name>,"fullname":<Course full name>}
        Response: {'status':status,'response':response}
        '''
        response=[]
        status="error"
        try:
            output={}
            courseinfodict={"courseid":courseid,"categoryid":categoryid,"fullname":fullname,"shortname":shortname}    
            print(courseinfodict)
            courseinfo=self._call(self.mWAP,'core_course_duplicate_course',**courseinfodict)
            #output['courseurl']=siteurl+'/course/view.php?id={}'.format(courseinfo['id'])
            status='success'
            response=courseinfo
        except:
            pass
        return {'status':status, 'response':response}

    def create_courses(self, datadicts, user_id=None):
        """
        Creates multiple Moodle courses using provided course metadata.

        This method accepts a list of course definition dictionaries,
        normalizes time fields (startdate, enddate), fills in default course 
        settings where needed, and invokes the Moodle web service API to create 
        the courses.

        Required fields in each course dictionary:
            - fullname (str): Full name of the course.
            - shortname (str): Short unique identifier for the course.
            - categoryid (int): ID of the Moodle course category.

        Optional fields that may be included in each course dictionary:
            - format (str): Course format (e.g., "topics", "weeks"). Default: "topics".
            - visible (int): Visibility status (1 = visible, 0 = hidden). Default: 1.
            - showgrades (int): Whether to show grades. Default: 0.
            - showreports (int): Whether to show activity reports. Default: 0.
            - groupmode (int): Default group mode. Default: 0.
            - groupmodeforce (int): Force group mode. Default: 0.
            - defaultgroupingid (int): Default grouping ID. Default: 0.
            - enablecompletion (int): Enable activity completion tracking. Default: 1.
            - completionnotify (int): Notify on activity completion. Default: 1.
            - startdate (str or int): Course start date, convertible to timestamp.
            - enddate (str or int): Course end date, convertible to timestamp.
            - summary (str): Course summary. Default: empty string.

        Parameters:
            datadicts (List[Dict]): A list of dictionaries containing course info.

        Returns:
            dict: {
                'status': 'success' or error message,
                'response': {
                    'meta_data': API response metadata or {},
                    'data': JSON string of data or '{}',
                    'message': Response message or error details
                }
            }
        
        Raises:
            Exception: If the API call to Moodle fails, an error message will be included in the response.
        """

        course={"fullname":"fullname",
                "shortname":"shortname", 
                "categoryid":0,
                "format":"topics", #course format: weeks, topics, social, site,..
                "summary":"",
                "showgrades":0, 
                "startdate":0,
                "enddate":0,
                "showreports":0,
                "visible":1,
                "groupmode":0,
                "groupmodeforce":0,
                "defaultgroupingid":0,
                "enablecompletion":1,
                "completionnotify":1}
        response=datadicts
        timeCols={"startdate","enddate"}
        tempDicts=[]
        for dct in datadicts:
            cols2change=list(set([*dct]).intersection(timeCols))
            if len(cols2change)!=0:
                for zz in cols2change:
                    try:
                        dct[zz]=int(time.mktime(to_datetime(dct[zz], errors='ignore', dayfirst=True).to_pydatetime().timetuple()))
                    except:
                        dct.pop(zz)
            tempDicts+=[dct]            
        courses=[]
        for dct in tempDicts:
            tmp=copy.deepcopy(course)
            for ky in [*dct]:
                tmp[ky]=dct[ky]
            courses+=[tmp]

        response=courses
        status="error"
        try:
            response=self._call(self.mWAP,'core_course_create_courses',courses=courses)
            status='success'
            
            if isinstance(response, list):
                meta_data = {'records':[
                    {**zz, "url": f"{moodleURL}course/view.php?id={zz['id']}"}
                    for zz in response
                ]}
                urls= [f"{moodleURL}course/view.php?id={zz['id']}" for zz in response]
                message = f"The following courses were created: {urls}"          
            else:
                meta_data = {'records':[]}  
                message = f"WS response: {response}"        
        except Exception as e:
            status = f"Error creating courses: {str(e)}"
            message = status
            data_json = '{}'
            meta_data = {}

        data_json = json.dumps(meta_data)
        response = {"meta_data": meta_data, "data":data_json, "message":message}    
        return {'status':status, 'response':response}

    def delete_courses(self,courseids, user_id=None):
        '''
        Webserice call: {"coursenames":<[list of course ids]>}
        Response: {'status':status,'response':response}
        '''
        response=[]
        status='error'
        #courseDicts=[]
        courseIDs=[]
        try:
            response=self._call(self.mWAP,'core_course_delete_courses',courseids=courseids)
            status='success'
        except:
            pass

        return {'status':status, 'response':response}

    def create_course_sections(self, datadicts, user_id=None):
        """
        Creates and updates multiple sections in Moodle courses via web service calls.

        This method performs two steps per course:
        1. Calls `local_wsmanagesections_create_sections` to create section stubs.
        2. Updates those sections via `local_wsmanagesections_update_sections` with names and visibility.

        Each dictionary in the input list must specify the course ID and the section's name and number.

        Expected dictionary format:
            {
                "courseid": <int>,         # Required. ID of the Moodle course.
                "sectionname": <str>,      # Required. Name/title of the section.
                "sectionnumber": <int>     # Required. Logical position/number of the section.
            }

        Parameters:
            datadicts (List[Dict]): A list of dictionaries where each dictionary contains metadata
                                    for a course section to be created and updated.

        Returns:
            dict: {
                'status': str,  # "success" if all course sections created and updated, otherwise error message.
                'response': {
                    'meta_data': list of created section metadata or list of error messages,
                    'data': JSON-encoded string version of meta_data,
                    'message': Description of what was created or error summary
                }
            }

        Raises:
            Exception: If any part of the course section creation or update fails, error info will
                    be embedded in the response under 'meta_data' and 'message'.
        """

        response=[]
        status="error"
        courseids=list(set([dct['courseid'] for dct in datadicts]))
        errors=[]
        meta_data=[]
        for courseid in courseids:
            sections=[]
            courseSecnDataDicts=[dct for dct in datadicts if dct['courseid']==courseid]
            parameters={"courseid":courseid,"position":0,"number":len(courseSecnDataDicts)}
            try:
                createdsections=self._call(self.mWAP,'local_wsmanagesections_create_sections',**parameters)
                if len(createdsections)!=0:
                    for iscn,csecn in enumerate(courseSecnDataDicts):
                        sections+=[{"type":"id","section":createdsections[iscn]['sectionid'],"name":csecn['sectionname'],"visible":1}]
                        response+=[{"courseid":courseid,'sectionid':createdsections[iscn]['sectionid'],"name":csecn['sectionname']}]
                    params={"courseid":courseid,"sections":sections}
                    response2=self._call(self.mWAP,'local_wsmanagesections_update_sections',**params)
                meta_data+=[{"courseid":courseid,'sectionid':zz['sectionid'],'sectionnumber':zz['sectionnumber']} for zz in createdsections]
            except Exception as e:
                errors.append(str(e))

        if errors:
            meta_data = errors
            status = f"Error creating courses: {errors}"
        else:
            status='success'
            message = f"The following sections were created: {meta_data}"
        data_json = json.dumps({"records":meta_data})
        response = {"meta_data": meta_data, "data":data_json, "message":message}                
        return {'status':status,'response':response}

    def update_course_sections(self,datadicts, user_id=None):
        '''
        Webserice call: datadicts=[{"courseid":<Course id>,"sectionid":<sectionid>,"name":<sectionname>,"visible":<visible>}]
        Response: {'status':status,'response':response}
        '''
        response=[]
        status="error"
        courseids=list(set([dct['courseid'] for dct in datadicts]))
    
        temp=[]
        for courseid in courseids:
            temp+=[{'courseid':courseid,'sections':[{'type':'id','section':dct['sectionid'],'name':dct['name'],'visible':dct['visible']} for dct in datadicts if dct['courseid']==courseid]}]
        
        for params in temp:
            response+=[self._call(self.mWAP,'local_wsmanagesections_update_sections',**params)]
        print(response)
        status='success'

        return {'status':status,'response':response}              

    def move_course_section(self,courseid,sectionid,position, user_id=None):
        '''
        Webserice call: {"courseid":<Course id>, "sectioninfo":{"type":"id","section":<sectionid>,"name":<sectionname>,"visible":<visible>}}
        Response: {'status':status,'response':response}
        '''
        response=[]
        status="error"
        try:
            self.engine.dispose()
            con=self.engine.connect()
            sectionnumber=read_sql(sql_text("SELECT section FROM mdl_course_sections WHERE id={}".format(sectionid)),con).to_dict(orient='records')[0]['section']
            params={"courseid":courseid,"sectionnumber":sectionnumber,"position":position}
            response=self._call(self.mWAP,'local_wsmanagesections_move_section',**params)
            print(response)
            status='success'
        except:
            pass
        con.close()
        return {'status':status,'response':response} 

    def delete_course_sections(self,courseid,sectionids, user_id=None):
        response=[]
        status='error'
        parameters={"courseid":courseid,"sectionids":sectionids}
        try:
            response=self._call(self.mWAP,'local_wsmanagesections_delete_sections',**parameters)
            status='success'
        except:
            pass

        return {'status':status, 'response':response}

######################################
#Module Creation/Update
######################################

    def get_modules_information(self, moduleids, user_id=None):
        response=[]
        status='error'
        self.engine.dispose()
        con=self.engine.connect()
        try:
            sqlQ="SELECT mcm.*, mm.name AS modulename FROM mdl_course_modules mcm INNER JOIN mdl_modules mm ON mm.id=mcm.module WHERE mcm.id IN {}".format(tuple(moduleids+[0]))
            response=read_sql(sql_text(sqlQ),con).to_dict(orient='records')
            status='success'
        except:
            pass

        con.close()
        return {'status':status,'response':response}

    def delete_course_modules(self,moduleids, user_id=None):
        response=[]
        status='error'
        #courseDicts=[]
        courseIDs=[]
        try:
            response=self._call(self.mWAP,'core_course_delete_modules',cmids=moduleids)
            status='success'
        except:
            pass

        return {'status':status, 'response':status}

    def create_course_module(self,modulename,courseid,sectionid,title,description="",templateModuleid=None,visible=1,optionsDict=None, user_id=None):
        response=[]
        status='error'
        if optionsDict==None:
            optionsDict={}

        if modulename=="questionnaire":
            options=self.get_questionnaire_options(**optionsDict)
        elif modulename=="assign":
            options=self.get_assign_options(**optionsDict)
        elif modulename=="forum":
            options=self.get_forum_options(**optionsDict)
        elif modulename=="schedulerr":
            options=self.get_scheduler_options(**optionsDict)
        elif modulename=="resource":
            options=self.get_resource_options(**optionsDict)
        elif modulename=="quiz":
            options=self.get_quiz_options(**optionsDict)
        elif modulename=="hvp":
            options=self.get_hvp_options(templateModuleid,**optionsDict)
        else:
            options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
        
        try:
            self.engine.dispose()
            con=self.engine.connect()
            sqlQs="SELECT section FROM mdl_course_sections WHERE id={}".format(sectionid)
            section=read_sql(sql_text(sqlQs),con)['section'].to_list()[0]
            
            #options=[{"name":"display", "value":1},{"name":"h5paction","value":"create"},{"name":"h5plibrary","value":libraryid},{"name":"params","value":json_content}]
            options=[{"name":"display","value":1}]+options
            fname="local_orisemodmanage_add_modules"
            modules=[{"modulename":modulename,"section":section,"name":title,"description":description,"descriptionformat":1,"visible":visible,"options":options}]
            kwargs={"courseid":courseid,"modules":modules}
            response=kwargs
            response=self._call(self.mWAP, fname, **kwargs)
            status="success"
        except:
           pass 
        con.close()                        
        return {"status":status,"response":response}

    def update_course_module(self,courseid,sectionid,moduleid,title=None,description=None,visible=1,optionsDict=None, user_id=None):
        response=[]
        status='error'
        # modulename=self.get_course_module_by_id(moduleid)['response'][0]['modname']
        # response=[{'mod':modulename}]
        self.engine.dispose()
        con=self.engine.connect()
        sqlQ = f"SELECT mm.name FROM mdl_course_modules mcm INNER JOIN mdl_modules mm ON mm.id=mcm.module WHERE mcm.id={moduleid}"
        modulename = read_sql(sql_text(sqlQ),con)['name'].to_list()[0]
        sqlQs="SELECT section FROM mdl_course_sections WHERE id={}".format(sectionid)
        section=read_sql(sql_text(sqlQs),con)['section'].to_list()[0]
        sqlQt="SELECT mh.name, mh.intro FROM mdl_{} mh INNER JOIN mdl_course_modules mcm ON mcm.instance=mh.id WHERE mcm.id={}".format(modulename,moduleid)
        modInfo=read_sql(sql_text(sqlQt),con)
        con.close()

        if optionsDict is None:
            optionsDict={}
        if title is None:
            title=modInfo['name'].to_list()[0]
        if description is None:
            description=modInfo['intro'].to_list()[0]    

        if modulename=="questionnaire":
            options=self.get_questionnaire_options(**optionsDict)
        elif modulename=="assign":
            options=self.get_assign_options(**optionsDict)
        elif modulename=="forum":
            options=self.get_forum_options(**optionsDict)
        elif modulename=="scheduler":
            options=self.get_scheduler_options(**optionsDict)
        elif modulename=="resource":
            options=self.get_resource_options(**optionsDict)
        elif modulename=="hvp":
            options=self.get_hvp_options(templateModuleid,**optionsDict)
        else:
            options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
        response=options
        fname="local_orisemodmanage_update_modules"
        #options=[{"name":"coursemodule","value":moduleid},{"name":"display","value":1}]+[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
        options=[{"name":"coursemodule","value":moduleid},{"name":"display","value":1}]+options
        modules=[{"modulename":modulename,"section":section,"name":title,"description":description,"descriptionformat":1,"visible":visible,"options":options}]
        kwargs={"courseid":courseid,"modules":modules}
        response=kwargs
        response=self._call(self.mWAP, fname, **kwargs)                         
        return {"status":status,"response":response}

    def get_questionnaire_options(self,showdescription=0,opendate=0,closedate=0,qtype=0,cannotchangerespondenttype=0,respondenttype="fullname",
                                resp_view=1,notifications=0,resume=0,navigate=0,autonum=3,progressbar=0,grade=0,
                                completionunlocked=1,completion=2,completionview=0,completionpass=0,completionexpected=0,completionsubmit=1, user_id=None):
        optionsDict={"showdescription":showdescription,
                "opendate":opendate,
                "closedate":closedate,
                "qtype":qtype,
                "cannotchangerespondenttype":cannotchangerespondenttype,
                "respondenttype":respondenttype,
                "resp_view":resp_view,
                "notifications":notifications,
                "resume":resume,
                "navigate":navigate,
                "autonum":autonum,
                "progressbar":progressbar,
                "grade": grade,
                "create":"new-0",
                "completionunlocked":completionunlocked,
                "completion":completion,"completionview":completionview,"completionpass":completionpass,
                "completionsubmit":completionsubmit,"completionexpected":completionexpected
                }
        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
        return options

    def get_quiz_options(self,showdescription = 1, intro="", introformat=1,
            timeopen=0,timeclose = 0,timelimit = 0, overduehandling = "autosubmit",graceperiod = 0,
            gradecat = 35,grade = 10, gradepass = 0, attempts = 0,grademethod = 1,
            questionsperpage = 1,navmethod = "free",shuffleanswers = 1,preferredbehaviour = "deferredfeedback",canredoquestions = 0,attemptonlast = 0,
            showuserpicture = 0,decimalpoints = 2,questiondecimalpoints = -1, showblocks = 0,
            seb_requiresafeexambrowser = 0,visibleoncoursepage = 1,groupmode = 0,groupingid = 0,
            completionunlocked = 1,completion = 0,completionattemptsexhausted = 0,completionminattempts = 0,completionexpected = 0,
            completionview = 0, completionpassgrade = 0, user_id=None):
        optionsDict={"showdescription":showdescription, "intro":intro, "introformat":introformat,
                     "timeopen":timeopen,"timeclose":timeclose,"timelimit":timelimit,"overduehandling":overduehandling,"graceperiod":graceperiod,
                    "gradecat":gradecat,"grade":grade, "gradepass":gradepass,"attempts":attempts,"grademethod":grademethod,
                    "questionsperpage":questionsperpage,"navmethod":navmethod,"shuffleanswers":shuffleanswers,"preferredbehaviour":preferredbehaviour,"canredoquestions":canredoquestions,"attemptonlast":attemptonlast,
                    "showuserpicture":showuserpicture,"decimalpoints":decimalpoints,"questiondecimalpoints":questiondecimalpoints, "showblocks":showblocks,
                    "seb_requiresafeexambrowser":seb_requiresafeexambrowser,"visibleoncoursepage":visibleoncoursepage,"groupmode":groupmode,"groupingid":groupingid,
                    "completionunlocked":completionunlocked,"completion":completion,"completionattemptsexhausted":completionattemptsexhausted,"completionminattempts":completionminattempts,"completionexpected":completionexpected,
                    "completionview":completionview, "completionpassgrade":completionpassgrade}
        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
        return options

    def get_forum_options(self,showdescription=0,type="general", duedate=0, cutoffdate=0, maxbytes=512000, 
                        maxattachments=9, displaywordcount=0, forcesubscribe=0, trackingtype=1, 
                        lockdiscussionafter=0, blockperiod=0, blockafter=0, warnafter=0, grade_forum=0, 
                        assessed=0, scale=100, gradepass=0,
                        completionpostsenabled=1,completionposts=1,completiondiscussions=0,completionreplies=0,
                        completionunlocked=1,completion=2,completionview=1,completionpass=0,completionexpected=0, user_id=None):
        optionsDict={"showdescription":showdescription,"type":type, "duedate":duedate, "cutoffdate":cutoffdate, "maxbytes":maxbytes, 
                        "maxattachments":maxattachments, "displaywordcount":displaywordcount, 
                        "forcesubscribe":forcesubscribe, "trackingtype":trackingtype, 
                        "lockdiscussionafter":lockdiscussionafter, "blockperiod":blockperiod, 
                        "blockafter":blockafter, "warnafter":warnafter, "grade_forum":grade_forum, 
                        "assessed":assessed, "scale":scale, "gradepass":gradepass,
                        "completionunlocked":completionunlocked,"completion":completion,"completionview":completionview,
                        "completionpostsenabled":completionpostsenabled, "completionposts":completionposts,"completiondiscussions":completiondiscussions,"completionreplies":completionreplies,
                        "completionpass":completionpass,"completionexpected":completionexpected}
        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
        return options

    def get_assign_options(self,showdescription=1,allowsubmissionsfromdate=0, duedate=0, cutoffdate=0, gradingduedate=0, 
                        assignsubmission_onlinetext_enabled=1, assignsubmission_file_enabled=1, 
                        assignsubmission_comments_enabled=1, assignsubmission_file_maxfiles=20, 
                        assignsubmission_file_maxsizebytes=0, assignfeedback_comments_enabled=1, 
                        assignfeedback_editpdf_enabled=1, assignfeedback_comments_commentinline=0, submissiondrafts=0, 
                        requiresubmissionstatement=0, maxattempts=-1, teamsubmission=0, 
                        preventsubmissionnotingroup=0, requireallteammemberssubmit=0, 
                        teamsubmissiongroupingid=0, sendnotifications=0, sendlatenotifications=0, 
                        sendstudentnotifications=1, grade=100, gradecat=5, 
                        gradepass=0, blindmarking=0, hidegrader=0, markingworkflow=0,
                        completionunlocked=1,completion=2,completionview=1,completionsubmit=1,completionusegrade=0,completionpass=0,completionexpected=0, user_id=None):

        optionsDict={"showdescription":showdescription,"allowsubmissionsfromdate":allowsubmissionsfromdate, "duedate":duedate, 
                        "cutoffdate":cutoffdate, "gradingduedate":gradingduedate, 
                        "assignsubmission_onlinetext_enabled":assignsubmission_onlinetext_enabled, 
                        "assignsubmission_file_enabled":assignsubmission_file_enabled, 
                        "assignsubmission_comments_enabled":assignsubmission_comments_enabled, 
                        "assignsubmission_file_maxfiles":assignsubmission_file_maxfiles, 
                        "assignsubmission_file_maxsizebytes":assignsubmission_file_maxsizebytes, 
                        "assignfeedback_comments_enabled":assignfeedback_comments_enabled, 
                        "assignfeedback_editpdf_enabled":assignfeedback_editpdf_enabled, 
                        "assignfeedback_comments_commentinline":assignfeedback_comments_commentinline, 
                        "submissiondrafts":submissiondrafts, "requiresubmissionstatement":requiresubmissionstatement, 
                        "maxattempts":maxattempts, "teamsubmission":teamsubmission, 
                        "preventsubmissionnotingroup":preventsubmissionnotingroup, 
                        "requireallteammemberssubmit":requireallteammemberssubmit, 
                        "teamsubmissiongroupingid":teamsubmissiongroupingid, "sendnotifications":sendnotifications, 
                        "sendlatenotifications":sendlatenotifications, 
                        "sendstudentnotifications":sendstudentnotifications, "grade":grade, "gradecat":gradecat, 
                        "gradepass":gradepass, "blindmarking":blindmarking, "hidegrader":hidegrader, 
                        "markingworkflow":markingworkflow,
                        "completionunlocked":completionunlocked,"completion":completion,
                        "completionusegrade":completionusegrade,"completionsubmit":completionsubmit,
                        "completionview":completionview,"completionpass":completionpass,"completionexpected":completionexpected}
        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]        
        return options   

    def get_scheduler_options(self,showdescription=0,mform_isexpanded_id_optionhdr=1,staffrolename="Facilitator", maxbookings=10, schedulermode="oneonly", 
                                bookingrouping=-1, guardtime=0, defaultslotduration=60, allownotifications=0, 
                                usenotes=1, grade=0, usebookingform=0, usestudentnotes=0, uploadmaxfiles=0, requireupload=0,
                                completionunlocked=1,completion=0,completionview=0,completionpass=0,completionexpected=0, user_id=None):

        optionsDict={"showdescription":showdescription,"mform_isexpanded_id_optionhdr":mform_isexpanded_id_optionhdr,"staffrolename":staffrolename, "maxbookings":maxbookings, "schedulermode":schedulermode, 
                    "bookingrouping":bookingrouping, "guardtime":guardtime, "defaultslotduration":defaultslotduration, 
                    "allownotifications":allownotifications, "usenotes":usenotes, "grade":grade, 
                    "usebookingform":usebookingform, "usestudentnotes":usestudentnotes, 
                    "uploadmaxfiles":uploadmaxfiles, "requireupload":requireupload,
                    "completionunlocked":completionunlocked,"completion":completion,"completionview":completionview,"completionpass":completionpass,"completionexpected":completionexpected
                    }

        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]        
        return options   
        
    def get_resource_options(self,showdescription=0,display=0,popupwidth=620, popupheight=450, printintro=1,filterfiles= 0,
                            completionunlocked=1,completion=2,completionview=1,completionpass=0,completionexpected=0, user_id=None):
        optionsDict={"showdescription":showdescription,"display":display,"popupwidth":popupwidth,"popupheight":popupheight,"printintro":printintro,"filterfiles":filterfiles,
                    "completionunlocked":completionunlocked,"completion":completion,"completionview":completionview,"completionpass":completionpass,"completionexpected":completionexpected}
        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]        
        return options

    def get_hvp_options(self,moduleid,showdescription=0, h5paction="create", h5pmaxscore=0, gradecat=5, maximumgrade=10,
                        completionunlocked=1,completion=2,completionview=1,completionusegrade=1,completionpass=0,completionexpected=0,params=None, user_id=None):
        paramsOld, title, machine_name, libraryid=self.get_hvp_template_content_moodle(moduleid)
        if params==None:
            params=paramsOld

        optionsDict={"showdescription":showdescription,
                    "h5paction":h5paction,
                    "h5plibrary":libraryid,
                    "h5pmaxscore":h5pmaxscore,
                    "gradecat":gradecat,
                    "gradepass":0,
                    "maximumgrade":maximumgrade,
                    "completionunlocked":completionunlocked,"completion":completion,
                    "completionview":completionview,"completionpass":completionpass,"completionusegrade":completionusegrade,"completionexpected":completionexpected,
                    "params":params}
        options=[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]        
        return options

    def get_hvp_template_content_moodle(self,moduleid, user_id=None):
        regex = re.compile(r'<[^>]+>')
        #try:
        self.engine.dispose()
        con=self.engine.connect()
        #sqlQm="SELECT mh.json_content, mh.name, ml.machine_name FROM mdl_hvp mh INNER JOIN mdl_course_modules mcm ON mh.id=mcm.instance INNER JOIN mdl_hvp_libraries ml ON ml.id=mh.main_library_id WHERE mcm.id={}".format(moduleid)
        sqlQm="SELECT mh.main_library_id, mh.name, mh.json_content, mhl.machine_name, mhl.major_version, mhl.minor_version, mhl.patch_version FROM mdl_hvp mh INNER JOIN mdl_course_modules mcm ON mcm.instance=mh.id INNER JOIN mdl_hvp_libraries mhl WHERE mcm.id={} AND mhl.id=mh.main_library_id".format(moduleid)
        hvpRecord=read_sql(sql_text(sqlQm),con).to_dict(orient="records")[0]
        contentsjson=hvpRecord['json_content']  #.replace("\\n","").replace("&nbsp;","").replace("<div>","").replace("<p>","").replace("</div>","").replace("</p>","").replace("<\\/p>","").replace("\\","")

        title=hvpRecord['name']
        machine_name=hvpRecord['machine_name']
        libraryid="{} {}.{}".format(hvpRecord['machine_name'],hvpRecord['major_version'],hvpRecord['minor_version'])
        #contentDict=json.loads(contentsjson)
        #except:
        #    pass
        con.close()
        return regex.sub('', contentsjson), title, machine_name, libraryid        

    def create_hvp_module(self,courseid,sectionid,templateModuleid,title,description="",visible=1,optionsDict=None,hvpmethod=None,hvpparameters=None, user_id=None):
        """
        Creates a new H5P activity in a Moodle course by cloning an existing template module
        and optionally applying transformation logic to the content (e.g., injecting quiz questions).

        This method performs a two-step operation:
        1. Calls `create_course_module()` to duplicate the base H5P template into the specified course and section.
        2. Calls `update_hvp_module()` to optionally update the cloned content using a specified transformation function.

        Parameters:
        ----------
        courseid : int
            The ID of the Moodle course in which to create the H5P activity.

        sectionid : int
            The ID of the course section where the activity should be placed.

        templateModuleid : int
            The ID of the existing H5P template module to duplicate.
            This template provides the base H5P structure (e.g., a multiple-choice question shell).

        title : str
            The name/title of the new H5P module activity.

        description : str, optional
            The activity description to be shown in Moodle. Defaults to an empty string.

        visible : int, optional
            Visibility status of the module (1 = visible, 0 = hidden). Defaults to 1.

        optionsDict : dict, optional
            Optional dictionary of parameters to pass into the module creation logic. 
            These may include things like initial H5P config, metadata, or frontend options.

        hvpmethod : str, optional
            The name of a local method (from `self.hvp_local_methods`) to apply for content transformation.
            For example, 'update_hvp_MCQ_questions'.

        hvpparameters : dict, optional
            Dictionary of parameters to be passed into the `hvpmethod` function.
            Used to dynamically modify the H5P content (e.g., inject quiz questions or custom config).

        Returns:
        -------
        dict
            A dictionary containing:
            - 'status': 'success' or 'error'
            - 'response': Moodle web service API response or error payload

        Example:
        -------
        result = self.create_hvp_module(
            courseid=10,
            sectionid=2,
            templateModuleid=42,
            title="My New Interactive Quiz",
            hvpmethod="update_hvp_MCQ_questions",
            hvpparameters={"questions": [...]}  # Your H5P-compatible question JSON
        )

        Notes:
        -----
        - This method assumes that `create_course_module()` and `update_hvp_module()` are valid internal 
        methods that handle Moodle’s backend operations via local APIs or web services.
        - If `hvpmethod` is not provided, the content will remain identical to the template.
        """

        modulename='hvp'
        statusResponse=self.create_course_module(modulename,courseid,sectionid,title,description,templateModuleid=templateModuleid,visible=visible,optionsDict=optionsDict)
        moduleid=statusResponse['response'][0]['moduleid']
        statusResponse=self.update_hvp_module(courseid,sectionid,moduleid,hvpmethod=hvpmethod,hvpparameters=hvpparameters)
        
        return statusResponse

    def update_hvp_module(self,courseid,sectionid,moduleid,title=None,description=None,visible=1,optionsDict=None,hvpmethod=None,hvpparameters=None, user_id=None):
        clean = re.compile(r'<[^>]+>')
        response=[]
        status='error'
        description=""
        self.engine.dispose()
        con=self.engine.connect()
        try:
            json_content, name, hvpMainMachineName, libraryid=self.get_hvp_template_content_moodle(moduleid)
            args=(json.loads(json_content),)
            if optionsDict==None:
                optionsDict={}

            if title!=None:
                name=title

            if hvpmethod!=None:
                updatedContentDict=self.hvp_function_call(self.hvp_local_methods[hvpmethod],*args,**hvpparameters)
                #.replace("\\n","").replace("&nbsp;","").replace("<div>","").replace("<p>","").replace("</div>","").replace("</p>","")
                json_content=clean.sub('',json.dumps(updatedContentDict))
            optionsDict['params']=json_content
            optionsDict["showdescription"]=1
            optionsDict["h5paction"]="create"
            optionsDict["h5plibrary"]=libraryid

            modulename='hvp'
            sqlQs="SELECT section FROM mdl_course_sections WHERE id={}".format(sectionid)
            section=read_sql(sql_text(sqlQs),con)['section'].to_list()[0]
            
            fname="local_orisemodmanage_update_modules"
            options=[{"name":"coursemodule","value":moduleid},{"name":"display","value":1}]+[{"name":ky, "value":optionsDict[ky]} for ky in [*optionsDict]]
            modules=[{"modulename":modulename,"section":section,"name":name,"description":description,"descriptionformat":1,"visible":visible,"options":options}]
            kwargs={"courseid":courseid,"modules":modules}
            response=kwargs
            response=self._call(self.mWAP, fname, **kwargs)
            status='success'
        except Exception as e:
            status=f"Error: {str(e)}" 
            raise RuntimeError(f"Failed to update HVP module: {e}") from e
        con.close()
        url=f"{moodleURL}mod/hvp/view.php?id={moduleid}"
        if isinstance(response, list) and isinstance(response[0],dict):
            response[0].update({'url':url})           
        return {"status":status,"response":response}

    def get_hvp_interactions(self,hvpmethod,moduleid, user_id=None):
        response=[]
        status='error'
        try:
            contentDict, titleold, hvpMainMachineName, libraryid=self.get_hvp_template_content_moodle(moduleid)
            args=(contentDict,)
            parameters={}
            response=self.hvp_function_call(self.hvp_local_methods[hvpmethod],*args,**parameters)
        except:
            pass 
        return {"status":status,"response":response}

    def get_hvp_module_content_moodle(self,moduleid, user_id=None):
        regex = re.compile(r'<[^>]+>')
        #try:
        self.engine.dispose()
        con=self.engine.connect()
        #sqlQm="SELECT mh.json_content, mh.name, ml.machine_name FROM mdl_hvp mh INNER JOIN mdl_course_modules mcm ON mh.id=mcm.instance INNER JOIN mdl_hvp_libraries ml ON ml.id=mh.main_library_id WHERE mcm.id={}".format(moduleid)
        sqlQm="SELECT mh.main_library_id, mh.name, mh.json_content, mhl.machine_name, mhl.major_version, mhl.minor_version, mhl.patch_version FROM mdl_hvp mh INNER JOIN mdl_course_modules mcm ON mcm.instance=mh.id INNER JOIN mdl_hvp_libraries mhl WHERE mcm.id={} AND mhl.id=mh.main_library_id".format(moduleid)
        hvpRecord=read_sql(sql_text(sqlQm),con).to_dict(orient="records")[0]
        contentsjson=hvpRecord['json_content']  #.replace("\\n","").replace("&nbsp;","").replace("<div>","").replace("<p>","").replace("</div>","").replace("</p>","").replace("<\\/p>","").replace("\\","")

        title=hvpRecord['name']
        machine_name=hvpRecord['machine_name']
        libraryid="{} {}.{}".format(hvpRecord['machine_name'],hvpRecord['major_version'],hvpRecord['minor_version'])
        #contentDict=json.loads(contentsjson)
        #except:
        #    pass
        con.close()
        status="success"
        response = regex.sub('', contentsjson)
        return {"status":status, "response":response}

##########
#HVP-MCQ
##########

    def add_multichoice_choice(self,text,correct, user_id=None):
        choice={'correct': correct, 'tipsAndFeedback': {'tip': '','chosenFeedback': '','notChosenFeedback': ''},'text': "{}".format(text)}
        return choice
    
    def update_hvp_MCQ_questions(self, *args, user_id=None, **kwargs):
        '''
        Webserice call: {"moduleid":<moduleid>}
        kwargs['questions']=[{'question':<>, 'answers':[{'text':<>,'correct':<1 or 0>}]}]
        Response: {'status':status,'response':response}
        ''' 
        contentDict=args[0]
        questions=kwargs['questions']
        temp=[]
        for intd in questions:
            question=copy.deepcopy(contentDict['questions'][0])
            question['params']['question']="{}".format(intd['question'])
            choices=[]
            for dct in intd['answers']:
                choices+=[self.add_multichoice_choice(dct['text'],dct['correct']==1)]
            question['params']['answers']=choices
            temp+=[question]                
        contentDict['questions']=copy.deepcopy(temp)
        # #self.add_hvp_template_content_dict(contentDict)
        response=contentDict
        #reponse=questions
        return response   

    def get_hvp_MCQ_questions(self,contentDict, user_id=None):
        '''
        Webserice call: {"moduleid":<moduleid>}
        questions=[{'question':<>, 'answers':[{'text':<>,'correct':<1 or 0>}]}]
        Response: {'status':status,'response':response}
        ''' 
        response={'questions':contentDict}
        status='error' 
        try:
            questions=contentDict['questions']
            temp=[]
            for intd in questions:
                tmpDct={}
                tmpDct['question']=intd['params']['question']
                choices=[]
                for dct in intd['params']['answers']:
                    choices+=[{'text':dct['text'],'correct':dct['correct']}]
                tmpDct['answers']=choices
                temp+=[tmpDct]                    
            response={'questions':temp}
            status='success'
        except:
           pass
        return response

    def create_hvp_MCQ_quiz(self,templatemoduleid,courseid,sectionid,title,questionsdict,optionsDict=None, user_id=None):
        """
        Creates a new H5P-based Multiple Choice Quiz (MCQ) in a specified Moodle course and section,
        using an existing H5P template module and populating it with the provided questions.

        The function builds an H5P-compatible data structure for a multiple-choice quiz, 
        where each question includes one correct answer and multiple incorrect ones.
        It then calls the internal `create_hvp_module` method to generate the H5P activity.

        Parameters:
        ----------
        templatemoduleid : int
            The ID of the H5P template module to use for generating the new quiz.
            This template typically contains the base structure of a multiple-choice H5P activity.

        courseid : int
            The Moodle course ID where the new H5P quiz will be created.

        sectionid : int
            The section ID within the course where the quiz should be placed.

        title : str
            The title of the quiz activity to be created in Moodle.

        questionsdict : list of dict
            A list of question dictionaries. Each dictionary must have the following structure:
            {
                'question': 'What is the capital of France?',
                'correct': 'Paris',
                'wrong1': 'Berlin',
                'wrong2': 'Madrid',
                ...
            }
            The keys `'wrong1'`, `'wrong2'`, etc. can be any keys except 'question' and 'correct'.
            Only non-empty incorrect options are included.

        Returns:
        -------
        dict
            A dictionary with the following structure:
            {
                'status': 'success' or 'error',
                'response': <API response data or intermediate payload>
            }

        Example:
        -------
        questions = [
            {'question': 'What is 2 + 2?', 'correct': '4', 'wrong1': '3', 'wrong2': '5'},
            {'question': 'Capital of Germany?', 'correct': 'Berlin', 'wrong1': 'Munich'}
        ]
        
        result = create_hvp_MCQ_quiz(
            templatemoduleid=42,
            courseid=10,
            sectionid=3,
            title="Basic Math & Geography Quiz",
            questionsdict=questions
        )

        Notes:
        -----
        - This function assumes that `create_hvp_module()` is a valid internal method
        that takes the given parameters and creates the H5P content in Moodle.
        - The structure of the H5P template module must be compatible with the 
        `update_hvp_MCQ_questions` method.
        - If `create_hvp_module` fails, the function will silently return status `'error'`.

        """
        response=[]
        status='error'
        templateModuleid=templatemoduleid
        hvpmethod="update_hvp_MCQ_questions"
        questions=[]
        for dct in questionsdict:
            questions+=[{'question':dct.pop('question'), 'answers':[{"text":dct.pop('correct'),'correct':1}]+[{'text':dct[ky],'correct':0} for ky in [*dct] if ky not in ['grade','name', 'label'] and dct[ky] not in ['', None]]}]

        hvpparameters={"questions":questions}
        response=hvpparameters
        #try:
        response=self.create_hvp_module(courseid,sectionid,templateModuleid,title,hvpmethod=hvpmethod,hvpparameters=hvpparameters, optionsDict=optionsDict)['response']
        status='success'
        #except:
        #   pass
        if isinstance(response, list) and response:
            meta_data = response[0]
        else:
            meta_data = {}
        message = f"Module created: {meta_data.get('url','')}"
        data_json = json.dumps(meta_data)
        response = {"meta_data": meta_data, "data":data_json, "message":message}                    
        return {"status":status,"response":response}

############################
#HVP - Speak the words set
############################

    def update_hvp_SW(self,*args, user_id=None,**kwargs):
        '''
        Webserice call: {"moduleid":<moduleid>}
        questions=[{"title":"Title Goes Here", "inputLanguage":"si-LK","question":"The question", "acceptedAnswers":["කුමක් ද"]}]
        Response: {'status':status,'response':response}
        ''' 
        contentDict=args[0]
        questions=kwargs['questions']
        temp=[]
        for qq in questions:
            question=copy.deepcopy(contentDict['questions'][0])
            question['metadata']['extraTitle']="{}".format(qq['title'])
            question['metadata']['title']="{}".format(qq['title'])
            question['params']['inputLanguage']=qq['inputLanguage']
            question['params']['question']=qq['question']
            question['params']['acceptedAnswers']=qq['acceptedAnswers']
            temp+=[question]                
        contentDict['questions']=copy.deepcopy(temp)
        #self.add_hvp_template_content_dict(contentDict)
        return  contentDict
            
    def get_hvp_SW_questions(self,contentDict, user_id=None):
        '''
        Webserice call: {"moduleid":<moduleid>}
        questions=[{"title":"Title Goes Here", "inputLanguage":"si-LK","question":"The question", "acceptedAnswers":"කුමක් ද"}]
        Response: {'status':status,'response':response}
        ''' 
        response={'questions':contentDict}
        status='error' 
        try:
            questions=contentDict['questions'] 
            temp=[]
            for intd in questions:
                tmpDct={}
                tmpDct['title']=intd['metadata']['title']
                tmpDct['inputLanguage']=intd['params']['inputLanguage']
                tmpDct['question']=intd['params']['question']
                tmpDct['acceptedAnswers']=intd['params']['acceptedAnswers']
                temp+=[tmpDct]                    
            response={'questions':temp}
            status='success'
        except:
           pass
        return response
    
############################
#HVP - Fill in the blanks
############################

    def update_hvp_fill_in_the_blanks(self,*args, user_id=None,**kwargs):
        '''
        Webserice call: {"moduleid":<moduleid>}
        text="Title Goes Here"
        questions=[list of strings with *blank*]
        Response: {'status':status,'response':response}
        ''' 
        contentDict=args[0]
        contentDict['text']= kwargs.get('text','Title')        
        contentDict['questions']=kwargs.get('questions',[]) 
        #self.add_hvp_template_content_dict(contentDict)
        return  contentDict
            
############################
#HVP - Drag the words
############################

    def update_hvp_drag_the_words(self,*args, user_id=None,**kwargs):
        '''
        Webserice call: {"moduleid":<moduleid>}
        taskDescription="Task description goes here"
        textField=Strings with *blank*
        Response: {'status':status,'response':response}
        ''' 
        
        contentDict=args[0]
        contentDict['taskDescription']= kwargs.get('taskDescription','Drag the words into the correct boxes')        
        contentDict['textField']=kwargs.get('textField','') 
        #self.add_hvp_template_content_dict(contentDict)
        return  contentDict

###################
#HVP Interactive Video
###################

    def add_video(self,contentDict,url, user_id=None):
        contentDict['interactiveVideo']['video']['files']=[{'path': url,'mime': 'video/YouTube','copyright': {'license': 'U'}}]
        self.add_hvp_template_content_dict(contentDict)
        return contentDict

    def update_hvp_videointeractions_MCQ(self,*args, user_id=None,**kwargs):
        '''
        Webserice call: {"moduleid":<moduleid>}
        interactions=[{'label':<>, 'question':<>, 'answers':[{'text':<>,'correct':<1 or 0>}], 'start':<>, 'end':<>}]
        Response: {'status':status,'response':response}
        ''' 
        contentDict=args[0]
        url=kwargs['url']
        interactions=kwargs['interactions']
        if url!='':
            contentDict['interactiveVideo']['video']['files']=[{'path': url,'mime': 'video/YouTube','copyright': {'license': 'U'}}]
        temp=[]
        for intd in interactions:
            interaction=copy.deepcopy(contentDict['interactiveVideo']['assets']['interactions'][0])
            interaction['action']['metadata']['extraTitle']="{}".format(intd['label'])
            interaction['action']['metadata']['title']="{}".format(intd['label'])
            interaction['label']="{}".format(intd['label'])
            interaction['duration']={'from':intd['start'],'to':intd['end']}
            interaction['action']['params']['question']="{}".format(intd['question'])
            choices=[]
            for dct in intd['answers']:
                choices+=[self.add_multichoice_choice(dct['text'],dct['correct']==1)]
            interaction['action']['params']['answers']=choices
            temp+=[interaction]                
        contentDict['interactiveVideo']['assets']['interactions']=copy.deepcopy(temp)
        #self.add_hvp_template_content_dict(contentDict)
        return  contentDict

    def get_hvp_video_interactions(self,contentDict, user_id=None):
        '''
        Webserice call: {"moduleid":<moduleid>}
        interactions=[{'label':<>, 'question':<>, 'answers':[{'text':<>,'correct':<1 or 0>}], 'duration':{'start':<>, 'end':<>}}]
        Response: {'status':status,'response':response}
        ''' 
        response={"files":[],"interactions":contentDict}
        status='error' 
        try:
            files=contentDict['interactiveVideo']['video']['files']
            interactions=contentDict['interactiveVideo']['assets']['interactions'] 
            temp=[]
            for intd in interactions:
                tmpDct={}
                tmpDct['label']=intd['label']
                tmpDct['duration']=intd['duration']
                tmpDct['question']=intd['action']['params']['question']
                choices=[]
                for dct in intd['action']['params']['answers']:
                    choices+=[{'text':dct['text'],'correct':dct['correct']}]
                tmpDct['answers']=choices
                temp+=[tmpDct]                    
            response={"files":files,"interactions":temp}
            status='success'
        except:
           pass
        return response

    def get_video_interactions_list(self, user_id=None):
        clean = re.compile(r'<[^>]+>')
        interactionsList={re.sub(clean, '', dct['label']):dct for dct in self.contentDict['interactiveVideo']['assets']['interactions']}
        return interactionsList

    def get_video_interaction(self,label, user_id=None):
        interaction=[dct for dct in self.contentDict['interactiveVideo']['assets']['interactions'] if re.sub(clean, '', dct['label'])==label][0]
        return interaction  

    def ai_generated_general_hvp_MCQ_quiz(self,numberofquestions,topic,templatemoduleid,courseid,sectionid,title, user_id=None):
        response=[]
        status='error'
        functionname='create_general_mcq_questions'
        parameters={"numberofquestions":numberofquestions,"topic":topic}
        try:
            questionsdict=json.loads(self.call_chatgpt_API(functionname,parameters,model="gpt-3.5-turbo", temperature=1.0,max_tokens=2000)['response'])
            response=self.create_hvp_MCQ_quiz(templatemoduleid,courseid,sectionid,title,questionsdict)['response']
            #response=questionsdict
            status='success'
        except:
            pass
        return {"status":status,"response":response}  

########################################################
#Moodle Analytics
########################################################

    def get_categories_course_modules_student_grades_competency_figures(self,categoryids, userids=None, user_id=None):
        """
        Retrieves and visualizes student competency attainment data for specified course categories.

        This method:
        - Executes a complex SQL query to collect final grade data, competency mappings, and activity-level info 
        for students enrolled in Moodle courses within the given `categoryids`.
        - Optionally filters results by a list of `userids`.
        - Computes normalized attainment metrics (`attainment`, `minattainment`, `maxattainment`) per competency.
        - Produces three Plotly visualizations:
            1. Program relative competency attainment (summed by competency)
            2. Maximum possible relative attainment
            3. Student-wise relative competency attainment by activity

        Parameters:
            categoryids (list[int]): A list of Moodle course category IDs to include in the analysis.
            userids (list[int], optional): Optional list of student user IDs to restrict the results to specific users.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if completed, or 'error' if an exception was raised.
                - 'response' (dict):
                    - 'meta_data' (str): Textual summary describing the operation performed.
                    - 'data' (str): JSON-encoded string containing:
                        - `records`: A list of student-competency-grade records.
                        - `figures`: A list of HTML strings for embedding Plotly charts.
                    - 'message' (str): Status message or error details.

        Output Records Include:
            - userid, firstname, lastname, fullname
            - competencyname, coursename, categoryname, sectionname, activityname
            - grademin, grademax, gradepass, finalgrade
            - attainment (normalized per student)
            - minattainment, maxattainment

        Notes:
            - Charts are rendered using Plotly (`plotly.express`) and embedded as HTML strings.
            - Attainment is normalized per student by dividing final grades by maximum grade and student count.
            - Filters for only non-deleted course modules and grade items where aggregation status is 'used'.

        Example:
            >>> get_category_competency_attainment_figures([3, 5], userids=[101, 102])
            {
                'status': 'success',
                'response': {
                    'meta_data': 'Retrieved detailed student grade data...',
                    'data': '{"records": [...], "figures": ["<div id=...>", ...]}',
                    'message': 'success'
                }
            }

        Raises:
            Exception: Any database or charting errors are caught and returned in the response.
        """
        fig0=go.Figure()
        figures=['','','']
        status='error'
        message=status
        self.engine.dispose()
        con=self.engine.connect()
        try:
            sqlQ="SELECT mgg.userid, mcomp.shortname AS competencyname, mu.firstname, mu.lastname, mcc.name AS categoryname, mc.shortname AS coursename, mcs.name AS sectionname, mgi.itemname AS activityname, mgi.grademin, mgi.grademax, mgi.gradepass, mgg.finalgrade FROM mdl_grade_items mgi INNER JOIN mdl_grade_grades mgg ON mgg.itemid=mgi.id INNER JOIN mdl_modules mm ON mm.name=mgi.itemmodule INNER JOIN mdl_course_modules mcm ON mcm.module=mm.id INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mgi.courseid INNER JOIN mdl_course_categories mcc on mcc.id=mc.category INNER JOIN mdl_user mu ON mu.id=mgg.userid INNER JOIN mdl_competency_modulecomp mcompm on mcompm.cmid=mcm.id INNER JOIN mdl_competency mcomp ON mcomp.id=mcompm.competencyid INNER JOIN mdl_competency_framework mcompf ON mcompf.id=mcomp.competencyframeworkid WHERE mcm.deletioninprogress=0 AND mcc.id IN {} AND mgi.itemtype='mod' AND mgg.aggregationstatus='used' AND mcm.instance=mgi.iteminstance".format(tuple(categoryids+[0]))
            if userids:
                sqlQ="SELECT mgg.userid, mcomp.shortname AS competencyname, mu.firstname, mu.lastname, mcc.name AS categoryname, mc.shortname AS coursename, mcs.name AS sectionname, mgi.itemname AS activityname, mgi.grademin, mgi.grademax, mgi.gradepass, mgg.finalgrade FROM mdl_grade_items mgi INNER JOIN mdl_grade_grades mgg ON mgg.itemid=mgi.id INNER JOIN mdl_modules mm ON mm.name=mgi.itemmodule INNER JOIN mdl_course_modules mcm ON mcm.module=mm.id INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mgi.courseid INNER JOIN mdl_course_categories mcc on mcc.id=mc.category INNER JOIN mdl_user mu ON mu.id=mgg.userid INNER JOIN mdl_competency_modulecomp mcompm on mcompm.cmid=mcm.id INNER JOIN mdl_competency mcomp ON mcomp.id=mcompm.competencyid INNER JOIN mdl_competency_framework mcompf ON mcompf.id=mcomp.competencyframeworkid WHERE mcm.deletioninprogress=0 AND mcc.id IN {} AND mgi.itemtype='mod' AND mgg.aggregationstatus='used' AND mcm.instance=mgi.iteminstance AND mgg.userid IN {}".format(tuple(categoryids+[0]),tuple(userids+[0]))
            responsePDF=read_sql(sql_text(sqlQ),con)
            totalStudents=len(list(set(responsePDF['userid'].to_list())))
            responsePDF['attainment']=(responsePDF['finalgrade']/responsePDF['grademax'])/totalStudents #(totalmaxgrades)
            responsePDF['minattainment']=(responsePDF['gradepass']/responsePDF['grademax'])/totalStudents #/totalActivities #/(totalStudents*totalActivities) #(totalStudents*totalActivities)
            responsePDF['maxattainment']=(responsePDF['grademax']/responsePDF['grademax'])/totalStudents #/(totalStudents*totalActivities) #/totalActivities #
            responsePDF['fullname']=responsePDF['firstname']+' '+responsePDF['lastname']

            fig = px.histogram(responsePDF, x='competencyname', y='attainment', histfunc='sum', title="Program relative competency attainement")
            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=True #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            figmax = px.histogram(responsePDF, x='competencyname', y='maxattainment', histfunc='sum', title="Program maximum possible relative competency attainment")
            plot_html2 = pio.to_html(
                figmax,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=True #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            figmax_return = plot_html2.replace('<div>', f'<div id="{fig_id}">')

            figstudent = px.bar(responsePDF, x='competencyname', y='attainment', color='fullname', text='activityname', title="Student wise relative competency attainement")
            plot_html3 = pio.to_html(
                figstudent,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=True #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            figstudent_return = plot_html3.replace('<div>', f'<div id="{fig_id}">')


            figures=[fig_return,figmax_return,figstudent_return]
            status='success'
            data_id=str(uuid.uuid4())
            message=f'''
            Retrieved detailed student grade data mapped to course competencies and generated related visualizations for categoryids={categoryids}:
            Columns: {', '.join(responsePDF.columns)}, data_id:{data_id}
            '''
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            data_id=None

        con.close()

        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":responsePDF.to_dict(orient="records"), "figures":figures})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response}

    def get_courses_modules_student_grades_competency_figures(self,courseids, userids=None, user_id=None):
        """
        Retrieves detailed student grade data mapped to course competencies and generates related visualizations.

        This method performs a complex SQL join across multiple Moodle tables to extract:
        - Student grades at the activity level
        - Competency mappings for course modules
        - Associated course and category metadata

        It also generates three Plotly figures:
        1. **Program relative competency attainment** (based on normalized final grades)
        2. **Program maximum possible relative competency attainment**
        3. **Student-wise relative competency attainment per competency**

        Parameters:
            courseids (list[int]): A list of course IDs to include in the query.
            userids (list[int], optional): Optional list of student user IDs to restrict the results to specific users.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if data was fetched and figures generated; otherwise, an error message.
                - 'response' (dict):
                    - 'meta_data' (str): Informational message or error.
                    - 'data' (str): JSON-encoded string of grade and competency data.
                    - 'figures' (list[str]): List of HTML strings for embedding Plotly figures in frontend.
                    - 'message' (str): Duplicate of `meta_data`, used for redundancy.

        Dataset Fields Returned:
            - userid (int): ID of the student.
            - firstname (str): Student's first name.
            - lastname (str): Student's last name.
            - fullname (str): Concatenated full name.
            - course name and category
            - section name
            - activity name and grading details (min, max, pass, final grade)
            - competencyname (str): Name of the competency the module is tied to.
            - attainment (float): Normalized score for the competency per student.
            - minattainment / maxattainment (float): Normalized minimum and max attainment levels.

        Notes:
            - Uses Plotly's `px.histogram` and `px.bar` to render visual summaries.
            - Normalizes attainment by dividing final grade by `grademax` and by total students.
            - Adds UUID-based div IDs to ensure charts are uniquely embedded.

        Example:
            >>> get_courses_modules_student_grades_competency_figures([101, 102],userids=[141,145])
            {
                'status': 'success',
                'response': {
                    'meta_data': 'success',
                    'data': '{"records": [...]}',
                    'figures': ['<div id="abcd1234">...</div>', '<div id="efgh5678">...</div>', '<div id="ijkl9012">...</div>'],
                    'message': 'success'
                }
            }

        Raises:
            Exception: Any issues during SQL execution, plotting, or data conversion are caught and included in the response.
        """
        response=[]
        status='error'
        self.engine.dispose()
        con=self.engine.connect()
        figures=['','','']
        try:
            sqlQ="SELECT mgg.userid, mcomp.shortname AS competencyname, mu.firstname, mu.lastname, mcc.name AS categoryname, mc.shortname AS coursename, mcs.name AS sectionname, mgi.itemname AS activityname, mgi.grademin, mgi.grademax, mgi.gradepass, mgg.finalgrade FROM mdl_grade_items mgi INNER JOIN mdl_grade_grades mgg ON mgg.itemid=mgi.id INNER JOIN mdl_modules mm ON mm.name=mgi.itemmodule INNER JOIN mdl_course_modules mcm ON mcm.module=mm.id INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mgi.courseid INNER JOIN mdl_course_categories mcc on mcc.id=mc.category INNER JOIN mdl_user mu ON mu.id=mgg.userid INNER JOIN mdl_competency_modulecomp mcompm on mcompm.cmid=mcm.id INNER JOIN mdl_competency mcomp ON mcomp.id=mcompm.competencyid INNER JOIN mdl_competency_framework mcompf ON mcompf.id=mcomp.competencyframeworkid WHERE mcm.deletioninprogress=0 AND mc.id IN {} AND mgi.itemtype='mod' AND mgg.aggregationstatus='used' AND mcm.instance=mgi.iteminstance".format(tuple(courseids+[0]))
            if userids:
                sqlQ="SELECT mgg.userid, mcomp.shortname AS competencyname, mu.firstname, mu.lastname, mcc.name AS categoryname, mc.shortname AS coursename, mcs.name AS sectionname, mgi.itemname AS activityname, mgi.grademin, mgi.grademax, mgi.gradepass, mgg.finalgrade FROM mdl_grade_items mgi INNER JOIN mdl_grade_grades mgg ON mgg.itemid=mgi.id INNER JOIN mdl_modules mm ON mm.name=mgi.itemmodule INNER JOIN mdl_course_modules mcm ON mcm.module=mm.id INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mgi.courseid INNER JOIN mdl_course_categories mcc on mcc.id=mc.category INNER JOIN mdl_user mu ON mu.id=mgg.userid INNER JOIN mdl_competency_modulecomp mcompm on mcompm.cmid=mcm.id INNER JOIN mdl_competency mcomp ON mcomp.id=mcompm.competencyid INNER JOIN mdl_competency_framework mcompf ON mcompf.id=mcomp.competencyframeworkid WHERE mcm.deletioninprogress=0 AND mc.id IN {} AND mgi.itemtype='mod' AND mgg.aggregationstatus='used' AND mcm.instance=mgi.iteminstance AND mgg.userid IN {}".format(tuple(courseids+[0]),tuple(userids+[0]))
            responsePDF=read_sql(sql_text(sqlQ),con)
            totalStudents=len(list(set(responsePDF['userid'].to_list())))
            responsePDF['attainment']=(responsePDF['finalgrade']/responsePDF['grademax'])/totalStudents #(totalmaxgrades)
            responsePDF['minattainment']=(responsePDF['gradepass']/responsePDF['grademax'])/totalStudents #/totalActivities #/(totalStudents*totalActivities) #(totalStudents*totalActivities)
            responsePDF['maxattainment']=(responsePDF['grademax']/responsePDF['grademax'])/totalStudents #/(totalStudents*totalActivities) #/totalActivities #
            responsePDF['fullname']=responsePDF['firstname']+' '+responsePDF['lastname']

            fig = px.histogram(responsePDF, x='competencyname', y='attainment', histfunc='sum', title="Program relative competency attainement")
            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=True #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            figmax = px.histogram(responsePDF, x='competencyname', y='maxattainment', histfunc='sum', title="Program maximum possible relative competency attainment")
            plot_html2 = pio.to_html(
                figmax,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=True #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            figmax_return = plot_html2.replace('<div>', f'<div id="{fig_id}">')

            figstudent = px.bar(responsePDF, x='competencyname', y='attainment', color='fullname', text='activityname', title="Student wise relative competency attainement")
            plot_html3 = pio.to_html(
                figstudent,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=True #'cdn'
            )
            fig_id = str(uuid.uuid4())[:8]
            figstudent_return = plot_html3.replace('<div>', f'<div id="{fig_id}">')

            figures=[fig_return,figmax_return,figstudent_return]
            status='success'
            data_id=str(uuid.uuid4())
            message=f'''
            Retrieved detailed student grade data mapped to course competencies and generated related visualizations for courseids={courseids}."
            Columns: {', '.join(responsePDF.columns)}, data_id:{data_id}
            '''
        except Exception as e:
            status=f'Error: {str(e)}'
            message=status
            data_id=None

        con.close()
        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":responsePDF.to_dict(orient="records"),"figures":figures})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status": status, "response": response}
       
    def get_grades_dedications_by_users_by_modules(self,userids, courseids=None, moduleids=None, user_id=None):
        response=[]
        status='error'
        contextlevel=70
        meta_data={}
        data_json='{}'
        message=''
        data_id=''
        records=[]
        self.engine.dispose()
        con=self.engine.connect()
        try:
            if moduleids:
                sqlQ="SELECT mctx.id AS contextid, mcm.score AS modulescore, mgg.finalgrade, mgg.rawgrade, mgg.rawgrademax, mgg.rawgrademin, mgi.itemname, mgi.grademax, mgi.grademin, mgi.gradepass, mu.firstname, mu.lastname, mc.shortname AS coursename, mcs.name AS sectionname, mm.name AS modulename, mcm.instance AS instanceid, mcm.section AS sectionid, mllmd.*, mllm.coursemoduleid AS moduleid, mllm.totaldedication AS moduletotaldedication, mllc.courseid, mllc.userid, mllc.totaldedication AS coursetotaldedication FROM mdl_local_ld_module_day mllmd INNER JOIN mdl_local_ld_module mllm ON mllm.id=mllmd.ldmoduleid INNER JOIN mdl_local_ld_course mllc ON mllc.id=mllm.ldcourseid INNER JOIN mdl_course_modules mcm ON mcm.id=mllm.coursemoduleid INNER JOIN mdl_modules mm ON mm.id=mcm.module INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mllc.courseid INNER JOIN mdl_user mu ON mu.id=mllc.userid INNER JOIN mdl_grade_items mgi ON mgi.iteminstance=mcm.instance INNER JOIN mdl_grade_grades mgg ON mgi.id=mgg.itemid INNER JOIN mdl_context mctx ON mctx.instanceid=mllm.coursemoduleid WHERE mgg.userid=mllc.userid AND mgi.itemmodule=mm.name AND mllm.coursemoduleid IN {} AND mllc.userid IN {} AND mcm.deletioninprogress=0 AND mctx.contextlevel={}".format(tuple(moduleids+[0]),tuple(userids+[0]),contextlevel)
            elif courseids:
                sqlQ="SELECT mctx.id AS contextid, mcm.score AS modulescore, mgg.finalgrade, mgg.rawgrade, mgg.rawgrademax, mgg.rawgrademin, mgi.itemname, mgi.grademax, mgi.grademin, mgi.gradepass, mu.firstname, mu.lastname, mc.shortname AS coursename, mcs.name AS sectionname, mm.name AS modulename, mcm.instance AS instanceid, mcm.section AS sectionid, mllmd.*, mllm.coursemoduleid AS moduleid, mllm.totaldedication AS moduletotaldedication, mllc.courseid, mllc.userid, mllc.totaldedication AS coursetotaldedication FROM mdl_local_ld_module_day mllmd INNER JOIN mdl_local_ld_module mllm ON mllm.id=mllmd.ldmoduleid INNER JOIN mdl_local_ld_course mllc ON mllc.id=mllm.ldcourseid INNER JOIN mdl_course_modules mcm ON mcm.id=mllm.coursemoduleid INNER JOIN mdl_modules mm ON mm.id=mcm.module INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mllc.courseid INNER JOIN mdl_user mu ON mu.id=mllc.userid INNER JOIN mdl_grade_items mgi ON mgi.iteminstance=mcm.instance INNER JOIN mdl_grade_grades mgg ON mgi.id=mgg.itemid INNER JOIN mdl_context mctx ON mctx.instanceid=mllm.coursemoduleid WHERE mgg.userid=mllc.userid AND mgi.itemmodule=mm.name AND mc.id IN {} AND mllc.userid IN {} AND mcm.deletioninprogress=0 AND mctx.contextlevel={}".format(tuple(courseids+[0]),tuple(userids+[0]),contextlevel)
            else:
                sqlQ="SELECT mctx.id AS contextid, mcm.score AS modulescore, mgg.finalgrade, mgg.rawgrade, mgg.rawgrademax, mgg.rawgrademin, mgi.itemname, mgi.grademax, mgi.grademin, mgi.gradepass, mu.firstname, mu.lastname, mc.shortname AS coursename, mcs.name AS sectionname, mm.name AS modulename, mcm.instance AS instanceid, mcm.section AS sectionid, mllmd.*, mllm.coursemoduleid AS moduleid, mllm.totaldedication AS moduletotaldedication, mllc.courseid, mllc.userid, mllc.totaldedication AS coursetotaldedication FROM mdl_local_ld_module_day mllmd INNER JOIN mdl_local_ld_module mllm ON mllm.id=mllmd.ldmoduleid INNER JOIN mdl_local_ld_course mllc ON mllc.id=mllm.ldcourseid INNER JOIN mdl_course_modules mcm ON mcm.id=mllm.coursemoduleid INNER JOIN mdl_modules mm ON mm.id=mcm.module INNER JOIN mdl_course_sections mcs ON mcs.id=mcm.section INNER JOIN mdl_course mc ON mc.id=mllc.courseid INNER JOIN mdl_user mu ON mu.id=mllc.userid INNER JOIN mdl_grade_items mgi ON mgi.iteminstance=mcm.instance INNER JOIN mdl_grade_grades mgg ON mgi.id=mgg.itemid INNER JOIN mdl_context mctx ON mctx.instanceid=mllm.coursemoduleid WHERE mgg.userid=mllc.userid AND mgi.itemmodule=mm.name AND mllc.userid IN {} AND mcm.deletioninprogress=0 AND mctx.contextlevel={}".format(tuple(userids+[0]),contextlevel)
            responsePDF=read_sql(sql_text(sqlQ),con)
            responsePDF['attainment']=responsePDF['finalgrade']/responsePDF['grademax']
            responsePDF['dedication']=responsePDF['dedication']/60.
            records=responsePDF.to_dict(orient="records")
            status="success"
            data_id=str(uuid.uuid4())
            message=f'''
            Retrieved student mudule grades and dedication data for moduleids={moduleids} and studentids={userids}."
            Columns: {', '.join(responsePDF.columns)}, data_id:{data_id}
            '''       
        except Exception as e:
            status='error'
            message = f'Error: {str(e)}'
        con.close()
        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":records})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status":status,"response":response} 

    def get_quiz_attempts_by_users(self,quizids=None, userids=None, user_id=None):
        response=[]
        status='error'
        contextlevel=70
        meta_data={}
        data_json='{}'
        message=''
        data_id=''
        records=[]
        self.engine.dispose()
        con=self.engine.connect()
        try:
            if userids:
                sqlQ="SELECT * FROM mdl_quiz_attempts mqa WHERE mqa.quiz IN {} mqa.userid IN {}".format(tuple(quizids+[0]),tuple(userids+[0]))
            else:
                sqlQ="SELECT * FROM mdl_quiz_attempts mqa WHERE mqa.quiz IN {}".format(tuple(quizids+[0]),tuple(userids+[0]))
            responsePDF=read_sql(sql_text(sqlQ),con)
            records=responsePDF.to_dict(orient="records")
            status="success"
            data_id=str(uuid.uuid4())
            message=f'''
            Retrieved quiz attempts for studentids={userids}."
            Columns: {', '.join(responsePDF.columns)}, data_id:{data_id}
            '''       
        except Exception as e:
            status='error'
            message = f'Error: {str(e)}'
        con.close()
        meta_data={"message":message, "data_id":data_id}
        data_json=json.dumps({"records":records})
        response = {"meta_data": meta_data, "data":data_json, "message":message}
        return {"status":status,"response":response} 

##########################################################################

    def build_h5p_mcq_quiz_json(self,questions, user_id=None):
        questions_input=questions
        base_ui = {
            "checkAnswerButton": "Check",
            "submitAnswerButton": "Submit",
            "showSolutionButton": "Show solution",
            "tryAgainButton": "Retry",
            "tipsLabel": "Show tip",
            "scoreBarLabel": "You got :num out of :total points",
            "tipAvailable": "Tip available",
            "feedbackAvailable": "Feedback available",
            "readFeedback": "Read feedback",
            "wrongAnswer": "Wrong answer",
            "correctAnswer": "Correct answer",
            "shouldCheck": "Should have been checked",
            "shouldNotCheck": "Should not have been checked",
            "noInput": "Please answer before viewing the solution",
            "a11yCheck": "Check the answers. The responses will be marked as correct, incorrect, or unanswered.",
            "a11yShowSolution": "Show the solution. The task will be marked with its correct solution.",
            "a11yRetry": "Retry the task. Reset all responses and start the task over again."
        }

        base_behaviour = {
            "enableRetry": False,
            "enableSolutionsButton": False,
            "enableCheckButton": True,
            "type": "auto",
            "singlePoint": False,
            "randomAnswers": True,
            "showSolutionsRequiresInput": False,
            "confirmCheckDialog": False,
            "confirmRetryDialog": False,
            "autoCheck": False,
            "passPercentage": 100,
            "showScorePoints": True
        }

        base_confirm_check = {
            "header": "Finish ?",
            "body": "Are you sure you wish to finish ?",
            "cancelLabel": "Cancel",
            "confirmLabel": "Finish"
        }

        base_confirm_retry = {
            "header": "Retry ?",
            "body": "Are you sure you wish to retry ?",
            "cancelLabel": "Cancel",
            "confirmLabel": "Confirm"
        }

        base_question_template = {
            "params": {
                "media": {"disableImageZooming": False, "type": {"params": {}}},
                "answers": [],
                "overallFeedback": [{"from": 0, "to": 100}],
                "behaviour": base_behaviour,
                "UI": base_ui,
                "confirmCheck": base_confirm_check,
                "confirmRetry": base_confirm_retry,
                "question": ""
            },
            "library": "H5P.MultiChoice 1.16",
            "metadata": {
                "contentType": "Multiple Choice",
                "license": "U",
                "title": "Untitled Multiple Choice",
                "authors": [],
                "changes": [],
                "extraTitle": "Untitled Multiple Choice"
            },
            "subContentId": ""
        }

        try:
            questions_output = []

            for idx, item in enumerate(questions_input):
                q = copy.deepcopy(base_question_template)
                q["params"]["question"] = item["question"]
                q["subContentId"] = str(uuid.uuid4())

                # Gather all answers
                answer_options = [item.get("correct")]
                for i in range(1, 5):
                    choice_key = f"choice{i}"
                    if item.get(choice_key) and item[choice_key] != item["correct"]:
                        answer_options.append(item[choice_key])

                # Remove duplicates
                answer_options = list(dict.fromkeys(answer_options))

                # Build answers with correct flag
                q["params"]["answers"] = [
                    {
                        "text": ans,
                        "correct": (ans == item["correct"]),
                        "tipsAndFeedback": {
                            "tip": "",
                            "chosenFeedback": "",
                            "notChosenFeedback": ""
                        }
                    }
                    for ans in answer_options
                ]

                questions_output.append(q)

            response = {
                "introPage": {
                    "showIntroPage": False,
                    "startButtonText": "Start Quiz",
                    "introduction": ""
                },
                "progressType": "dots",
                "passPercentage": 50,
                "questions": questions_output,
                "disableBackwardsNavigation": False,
                "randomQuestions": True,
                "endGame": {
                    "showResultPage": True,
                    "showSolutionButton": False,
                    "showRetryButton": True,
                    "noResultMessage": "Finished",
                    "message": "Your result:",
                    "scoreBarLabel": "You got @finals out of @totals points",
                    "overallFeedback": [{"from": 0, "to": 100}],
                    "solutionButtonText": "Show solution",
                    "retryButtonText": "Retry",
                    "finishButtonText": "Finish",
                    "submitButtonText": "Submit",
                    "showAnimations": False,
                    "skippable": False,
                    "skipButtonText": "Skip video"
                },
                "override": {"checkButton": False},
                "texts": {
                    "prevButton": "Previous question",
                    "nextButton": "Next question",
                    "finishButton": "Finish",
                    "submitButton": "Submit",
                    "textualProgress": "Question: @current of @total questions",
                    "jumpToQuestion": "Question %d of %total",
                    "questionLabel": "Question",
                    "readSpeakerProgress": "Question @current of @total",
                    "unansweredText": "Unanswered",
                    "answeredText": "Answered",
                    "currentQuestionText": "Current question",
                    "navigationLabel": "Questions"
                }
            }
            status="success"
        except Exception as e:
            status=f"Error: {str(e)}"
            response={}

        return {"status":status, "response":response}

    def save_h5p_quiz_from_records(self, quiz_type, title, grade, records, keywords=None, description=None, user_id=None):
        try:
            obj, created = MoodleQuizQuestions.objects.update_or_create(
                title=title,
                defaults={
                    "grade": grade,
                    "quiz_type":quiz_type,
                    "records": records,
                    "keywords": keywords or [],
                    "description": description or ""
                    # Don't pass "ref" here unless you want to overwrite it on every update ❗
                }
            )

            # Set ref only if created and it's missing
            if created and not obj.ref:
                obj.ref = uuid.uuid4().hex
                obj.save(update_fields=["ref"])

            status = "success"
            relative_url=''
            if obj.quiz_type == QuizChoices.MCQ.value:
                relative_url = reverse('h5p_mcq_quiz', kwargs={'ref': obj.ref})  # or kwargs={'pk': 123}
            if obj.quiz_type == QuizChoices.SPEAK_THE_WORD.value:
                relative_url = reverse('h5p_sw_quiz', kwargs={'ref': obj.ref})  # or kwargs={'pk': 123}
            
            view_url = f"https://{domain}{relative_url}"
            meta_data={
                    "id": obj.id,
                    "ref": obj.ref,
                    "url":view_url
                }
            response = {
                "meta_data": meta_data,
                "data":json.dumps(meta_data),
                "message": f"The saved quiz can be viewd at: {view_url}"
            }

        except Exception as e:
            status = f"error"
            response = {
                "message": str(e)
            }

        return {"status": status, "response": response}

    def save_h5p_quiz_from_data_id(self, quiz_type, title, grade, data_id, keywords=None, description=None, user_id=None):
        try:
            data_id = uuid.UUID(data_id)  # ensure it's a valid UUID
            saved_data = AiAssistantCallData.objects.get(data_id=data_id)

            obj, created = MoodleQuizQuestions.objects.update_or_create(
                title=title,
                defaults={
                    "grade": grade,
                    "quiz_type":quiz_type,
                    "records": json.loads(saved_data.records),
                    "keywords": keywords or [],
                    "description": description or ""
                }
            )

            # Set ref only if created and it's missing
            if created and not obj.ref:
                obj.ref = uuid.uuid4()
                obj.save(update_fields=["ref"])

            status = "success"
            relative_url=''
            if obj.quiz_type == QuizChoices.MCQ.value:
                relative_url = reverse('h5p_mcq_quiz', kwargs={'ref': obj.ref})  # or kwargs={'pk': 123}
            if obj.quiz_type == QuizChoices.SPEAK_THE_WORD.value:
                relative_url = reverse('h5p_sw_quiz', kwargs={'ref': obj.ref})  # or kwargs={'pk': 123}
            
            view_url = f"https://{domain}{relative_url}"
            meta_data={
                    "id": obj.id,
                    "ref": obj.ref,
                    "url":view_url
                }
            response = {
                "meta_data": meta_data,
                "data":json.dumps(meta_data),
                "message": f"The saved quiz can be viewd at: {view_url}"
            }

        except Exception as e:
            status = f"error"
            response = {
                "message": str(e)
            }

        return {"status": status, "response": response}

    def build_h5p_sw_quiz_json(self, questions, user_id=None):
        base_l10n_question = {
            "retryLabel": "Retry",
            "showSolutionLabel": "Show solution",
            "speakLabel": "Push to speak",
            "listeningLabel": "Listening...",
            "correctAnswersText": "The correct answer(s):",
            "userAnswersText": "Your answer(s) was interpreted as:",
            "noSound": "I could not hear you, make sure your microphone is enabled",
            "unsupportedBrowserHeader": "It looks like your browser does not support speech recognition",
            "unsupportedBrowserDetails": "Please try again in a browser like Chrome",
            "a11yShowSolution": "Show the solution. The task will be marked with its correct solution.",
            "a11yRetry": "Retry the task. Reset all responses and start the task over again."
        }

        base_l10n_main = {
            "introductionButtonLabel": "Start Course!",
            "solutionScreenResultsLabel": "Your results:",
            "showSolutionsButtonLabel": "Show solution",
            "retryButtonLabel": "Retry",
            "nextQuestionAriaLabel": "Next question",
            "previousQuestionAriaLabel": "Previous question",
            "navigationBarTitle": "Slide :num",
            "answeredSlideAriaLabel": "Answered",
            "activeSlideAriaLabel": "Currently active"
        }

        base_question_template = {
            "params": {
                "media": {
                    "disableImageZooming": False,
                    "type": {
                        "params": {}
                    }
                },
                "incorrectAnswerText": "Incorrect answer",
                "correctAnswerText": "Correct answer",
                "inputLanguage": "",  # will be set per question
                "l10n": base_l10n_question,
                "question": "",
                "acceptedAnswers": []
            },
            "library": "H5P.SpeakTheWords 1.5",
            "subContentId": "",
            "metadata": {
                "contentType": "Speak the Words",
                "license": "U",
                "title": "Untitled Speak Task",
                "authors": [],
                "changes": [],
                "extraTitle": ""
            }
        }

        try:
            questions_output = []

            for item in questions:
                q = copy.deepcopy(base_question_template)
                q["params"]["question"] = item["question"]
                q["params"]["acceptedAnswers"] = item["acceptedAnswers"]
                q["params"]["inputLanguage"] = item.get("inputLanguage", "en")  # default to English
                q["subContentId"] = str(uuid.uuid4())
                q["metadata"]["title"] = item.get("title", "Untitled")
                q["metadata"]["extraTitle"] = item.get("title", "")

                questions_output.append(q)

            response = {
                "introduction": {
                    "showIntroPage": False
                },
                "questions": questions_output,
                "overallFeedback": [{"from": 0, "to": 100}],
                "l10n": base_l10n_main
            }

            return {
                "status": "success",
                "response": response
            }

        except Exception as e:
            return {
                "status": f"error: {str(e)}",
                "response": {}
            }

#########################################

    def download_and_extract_library(self, machine_name, version, libraries_dir, user_id=None):
        """
        Download and extract H5P library if not already present.
        """
        lib_folder = libraries_dir / f"{machine_name}-{version}"
        if lib_folder.exists():
            return  # Already downloaded

        # URL mappings for public H5P libraries - Librarie download does not work..
        url_map = {
            "H5P.MultiChoice-1.16": "https://h5p.org/sites/default/files/h5p/libraries/H5P.MultiChoice-1.16.zip",
            "H5P.SpeakTheWords-1.5": "https://h5p.org/sites/default/files/h5p/libraries/H5P.SpeakTheWords-1.5.zip",
        }


        key = f"{machine_name}-1.{version}"  # include majorVersion
        if key not in url_map:
            raise ValueError(f"No known download URL for library: {key}")


        zip_url = url_map[key]
        zip_path = libraries_dir / f"{key}.zip"

        # Download zip
        resp = requests.get(zip_url)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(libraries_dir)

        # Remove zip
        zip_path.unlink()

    def build_h5p_export_file(self, ref, quiz_type, output_dir="media/exports", user_id=None):
        """
        Create a downloadable .h5p export package for a given MoodleQuizQuestions object.
        """
        try:
            quiz_obj = get_object_or_404(MoodleQuizQuestions, ref=ref, quiz_type=quiz_type)
            records = quiz_obj.records
            title = quiz_obj.title or "Untitled"

            # 📚 Select appropriate builder and library
            if quiz_type == QuizChoices.MCQ.value:
                response = self.build_h5p_mcq_quiz_json(questions=records)
                main_library = "H5P.MultiChoice"
                minor_version = 16
            elif quiz_type == QuizChoices.SPEAK_THE_WORD.value:
                response = self.build_h5p_sw_quiz_json(questions=records)
                main_library = "H5P.SpeakTheWords"
                minor_version = 5
            else:
                raise ValueError("Unsupported quiz type")

            quiz_data = response.get("response", {})

            # 🧱 Build paths
            quiz_uuid = str(uuid.uuid4())
            build_dir = Path(output_dir) / f"temp_h5p_{quiz_uuid}"
            content_dir = build_dir / "content"
            libraries_dir = build_dir / "libraries"
            build_dir.mkdir(parents=True, exist_ok=True)
            content_dir.mkdir(exist_ok=True)
            libraries_dir.mkdir(exist_ok=True)

            # 🧠 Write content.json
            with open(content_dir / "content.json", "w", encoding="utf-8") as f:
                json.dump(quiz_data, f, indent=2, ensure_ascii=False)

            # 🧠 Write h5p.json
            h5p_json = {
                "title": title,
                "language": "en",
                "mainLibrary": main_library,
                "embedTypes": ["div"],
                "license": "U",
                "preloadedDependencies": [
                    {
                        "machineName": main_library,
                        "majorVersion": 1,
                        "minorVersion": minor_version
                    }
                ]
            }
            with open(build_dir / "h5p.json", "w", encoding="utf-8") as f:
                json.dump(h5p_json, f, indent=2)

            # 🔽 Download & extract the library
            self.download_and_extract_library(main_library, minor_version, libraries_dir)

            # 🗜️ Zip to .h5p
            output_path = Path(output_dir) / f"{ref}.h5p"
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(build_dir):
                    for file in files:
                        filepath = Path(root) / file
                        arcname = filepath.relative_to(build_dir)
                        zipf.write(filepath, arcname)

            # 🧹 Cleanup
            shutil.rmtree(build_dir)

            meta_data = {"filepath": str(output_path)}
            response = {
                "meta_data": meta_data,
                "data": json.dumps(meta_data),
                "message": f"H5P file generated: {output_path}"
            }
            status = "success"
        except Exception as e:
            response = {
                "meta_data": {},
                "data": json.dumps({}),
                "message": f"H5P generation error: {str(e)}"
            }
            status = f"error: {str(e)}"

        return {"status": status, "response": response}

##########################################

    def create_hvp_MCQ_quiz_from_data_id(self,templatemoduleid,courseid,sectionid,title,data_id, quiz_type="mcq", user_id=None):
        """
        Creates a new H5P-based Multiple Choice Quiz (MCQ) in a specified Moodle course and section,
        using an existing H5P template module and populating it with the provided questions.

        The function builds an H5P-compatible data structure for a multiple-choice quiz, 
        where each question includes one correct answer and multiple incorrect ones.
        It then calls the internal `create_hvp_module` method to generate the H5P activity.

        Parameters:
        ----------
        templatemoduleid : int
            The ID of the H5P template module to use for generating the new quiz.
            This template typically contains the base structure of a multiple-choice H5P activity.

        courseid : int
            The Moodle course ID where the new H5P quiz will be created.

        sectionid : int
            The section ID within the course where the quiz should be placed.

        title : str
            The title of the quiz activity to be created in Moodle.

        data_id : A UUID of the saved quiz

        Returns:
        -------
        dict
            A dictionary with the following structure:
            {
                'status': 'success' or 'error',
                'response': <API response data or intermediate payload>
            }

        Example:
        -------
        
        result = create_hvp_MCQ_quiz(
            templatemoduleid=42,
            courseid=10,
            sectionid=3,
            title="Basic Math & Geography Quiz",
            data_id=uuid
        )

        Notes:
        -----
        - This function assumes that `create_hvp_module()` is a valid internal method
        that takes the given parameters and creates the H5P content in Moodle.
        - The structure of the H5P template module must be compatible with the 
        `update_hvp_MCQ_questions` method.
        - If `create_hvp_module` fails, the function will silently return status `'error'`.

        """
        response=[]
        status='error'
        templateModuleid=templatemoduleid
        try:
            data_id = uuid.UUID(data_id)  # ensure it's a valid UUID
            saved_data = AiAssistantCallData.objects.get(data_id=data_id)        

            if isinstance(saved_data.records, str):
                try:
                    questionsdict = json.loads(saved_data.records)
                except json.JSONDecodeError:
                    questionsdict = []
            elif isinstance(saved_data.records, list):
                questionsdict = saved_data.records
            else:
                questionsdict = []

            if quiz_type=="mcq":
                hvpmethod="update_hvp_MCQ_questions"
                questions=[]
                for dct in questionsdict:
                    questions+=[{'question':dct.pop('question'), 'answers':[{"text":dct.pop('correct'),'correct':1}]+[{'text':dct[ky],'correct':0} for ky in [*dct] if ky not in ['grade','name', 'label'] and dct[ky] not in ['', None]]}]
            elif quiz_type=="sw":
                hvpmethod="update_hvp_SW"
                questions = questionsdict


            hvpparameters={"questions":questions}
            response=hvpparameters
            response=self.create_hvp_module(courseid,sectionid,templateModuleid,title,hvpmethod=hvpmethod,hvpparameters=hvpparameters, optionsDict=None)['response']
            status='success'
            if isinstance(response, list) and response:
                meta_data = response[0]
            else:
                meta_data = {}
            message = f"Module created: {meta_data.get('url','')}"
        except Exception as e:
            status=f"Error: {str(e)}"
            meta_data ={}
            message=status

        data_json = json.dumps(meta_data)
        response = {"meta_data": meta_data, "data":data_json, "message":message}                    
        return {"status":status,"response":response}

    def create_hvp_quiz_from_quiz_ref(self,quiz_ref,templatemoduleid,courseid,sectionid,title, quiz_type="mcq", user_id=None):
        """
        Creates a new H5P-based Multiple Choice Quiz (MCQ) in a specified Moodle course and section,
        using an existing H5P template module and populating it with the provided questions.

        The function builds an H5P-compatible data structure for a multiple-choice quiz, 
        where each question includes one correct answer and multiple incorrect ones.
        It then calls the internal `create_hvp_module` method to generate the H5P activity.

        Parameters:
        ----------
        templatemoduleid : int
            The ID of the H5P template module to use for generating the new quiz.
            This template typically contains the base structure of a multiple-choice H5P activity.

        courseid : int
            The Moodle course ID where the new H5P quiz will be created.

        sectionid : int
            The section ID within the course where the quiz should be placed.

        title : str
            The title of the quiz activity to be created in Moodle.

        data_id : A UUID of the saved quiz

        Returns:
        -------
        dict
            A dictionary with the following structure:
            {
                'status': 'success' or 'error',
                'response': <API response data or intermediate payload>
            }

        Example:
        -------
        
        result = create_hvp_MCQ_quiz(
            templatemoduleid=42,
            courseid=10,
            sectionid=3,
            title="Basic Math & Geography Quiz",
            data_id=uuid
        )

        Notes:
        -----
        - This function assumes that `create_hvp_module()` is a valid internal method
        that takes the given parameters and creates the H5P content in Moodle.
        - The structure of the H5P template module must be compatible with the 
        `update_hvp_MCQ_questions` method.
        - If `create_hvp_module` fails, the function will silently return status `'error'`.

        """
        response=[]
        status='error'
        templateModuleid=templatemoduleid
        try:
            quiz_ref = uuid.UUID(quiz_ref)  # ensure it's a valid UUID
            saved_data = MoodleQuizQuestions.objects.get(ref=quiz_ref)        

            if isinstance(saved_data.records, str):
                try:
                    questionsdict = json.loads(saved_data.records)
                except json.JSONDecodeError:
                    questionsdict = []
            elif isinstance(saved_data.records, list):
                questionsdict = saved_data.records
            else:
                questionsdict = []

            if quiz_type=="mcq":
                hvpmethod="update_hvp_MCQ_questions"
                questions=[]
                for dct in questionsdict:
                    questions+=[{'question':dct.pop('question'), 'answers':[{"text":dct.pop('correct'),'correct':1}]+[{'text':dct[ky],'correct':0} for ky in [*dct] if ky not in ['grade','name', 'label'] and dct[ky] not in ['', None]]}]
            elif quiz_type=="sw":
                hvpmethod="update_hvp_SW"
                questions = questionsdict


            hvpparameters={"questions":questions}
            response=hvpparameters
            response=self.create_hvp_module(courseid,sectionid,templateModuleid,title,hvpmethod=hvpmethod,hvpparameters=hvpparameters, optionsDict=None)['response']
            status='success'
            if isinstance(response, list) and response:
                meta_data = response[0]
            else:
                meta_data = {}
            message = f"Module created: {meta_data.get('url','')}"
        except Exception as e:
            status=f"Error: {str(e)}"
            meta_data ={}
            message=status

        data_json = json.dumps(meta_data)
        response = {"meta_data": meta_data, "data":data_json, "message":message}                    
        return {"status":status,"response":response}  

    def create_hvp_quiz_from_quiz_ref_public(self,quiz_ref,title, quiz_type="mcq", courseid=None, sectionid=None, user_id=None):
        """
        Creates a new H5P-based Multiple Choice Quiz (MCQ) in a specified Moodle course and section,
        using an existing H5P template module and populating it with the provided questions.

        The function builds an H5P-compatible data structure for a multiple-choice quiz, 
        where each question includes one correct answer and multiple incorrect ones.
        It then calls the internal `create_hvp_module` method to generate the H5P activity.

        Parameters:
        ----------
        templatemoduleid : int
            The ID of the H5P template module to use for generating the new quiz.
            This template typically contains the base structure of a multiple-choice H5P activity.

        courseid : int
            The Moodle course ID where the new H5P quiz will be created.

        sectionid : int
            The section ID within the course where the quiz should be placed.

        title : str
            The title of the quiz activity to be created in Moodle.

        data_id : A UUID of the saved quiz

        Returns:
        -------
        dict
            A dictionary with the following structure:
            {
                'status': 'success' or 'error',
                'response': <API response data or intermediate payload>
            }

        Example:
        -------
        
        result = create_hvp_MCQ_quiz(
            templatemoduleid=42,
            courseid=10,
            sectionid=3,
            title="Basic Math & Geography Quiz",
            data_id=uuid
        )

        Notes:
        -----
        - This function assumes that `create_hvp_module()` is a valid internal method
        that takes the given parameters and creates the H5P content in Moodle.
        - The structure of the H5P template module must be compatible with the 
        `update_hvp_MCQ_questions` method.
        - If `create_hvp_module` fails, the function will silently return status `'error'`.

        """
        response=[]
        status='error'
        mcq_templatemoduleid=36
        sw_templatemoduleid=39
        if courseid is None:
            courseid = demo_courseid
            sectionid = demo_sectionid
            
        try:
            if quiz_type=='mcq':
                templatemoduleid=mcq_templatemoduleid
            elif quiz_type=='sw':
                templatemoduleid=sw_templatemoduleid
            response_dict = self.create_hvp_quiz_from_quiz_ref(quiz_ref,templatemoduleid,courseid,sectionid,title, quiz_type=quiz_type)
            meta_data = response_dict.get('response',{}).get('meta_data',{})
            url= meta_data.get('url','')        
            status='success'
            message = f"Module created: {url}"
        except Exception as e:
            status=f"Error: {str(e)}"
            meta_data ={}
            message=status

        data_json = json.dumps(meta_data)
        response = {"meta_data": meta_data, "data":data_json, "message":message}                    
        return {"status":status,"response":response}              