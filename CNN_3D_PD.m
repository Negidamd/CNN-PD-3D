function varargout = CNN_3D_PD(varargin)
% CNN_3D_PD MATLAB code for CNN_3D_PD.fig
%      CNN_3D_PD, by itself, creates a new CNN_3D_PD or raises the existing
%      singleton*.
%
%      H = CNN_3D_PD returns the handle to a new CNN_3D_PD or the handle to
%      the existing singleton*.
%
%      CNN_3D_PD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CNN_3D_PD.M with the given input arguments.
%
%      CNN_3D_PD('Property','Value',...) creates a new CNN_3D_PD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CNN_3D_PD_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CNN_3D_PD_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CNN_3D_PD

% Last Modified by GUIDE v2.5 29-Oct-2023 19:06:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CNN_3D_PD_OpeningFcn, ...
                   'gui_OutputFcn',  @CNN_3D_PD_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CNN_3D_PD is made visible.
function CNN_3D_PD_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CNN_3D_PD (see VARARGIN)

% Choose default command line output for CNN_3D_PD
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CNN_3D_PD wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CNN_3D_PD_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
        [handles.file,handles.path] = uigetfile({'*.nii','Nifti Files';'*.*','All Files' },'mytitle', ' ');
        if isequal(handles.file,0)
        disp('User selected Cancel');
        else
        disp(['User selected ', fullfile(handles.path,handles.file)]);
        end
        load '3DPretrainedModel_1.mat'
        inputSize = mriNet.Layers(1).InputSize;
        srcDir_Test   = {fullfile(handles.path,handles.file)};
        imds = imageDatastore(srcDir_Test{1}, 'FileExtensions', '.nii',...
        'ReadFcn',@niftiread,'ReadSize', 10);
        augimdsTest = transform(imds,@(data)classification3DAugmentationPipeline1(data,inputSize,'test'));
        [~,probs] = classify(mriNet,augimdsTest);
        str = "This image indicated " + round(100*probs(1,2),2) + "% probability of having PD."; 
        set(handles.text15, 'string', str)