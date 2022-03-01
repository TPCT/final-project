<?php

namespace controllers\embedded;

use core\MiddleWare;
use core\Request;
use core\Response;
use helpers\Helper;
use helpers\SecurityHelper;
use middlewares\EncryptionLayerMiddleWare;
use middlewares\TrainMiddleWare;
use models\embedded\ApiLogEventModel;

class ApiLogEventController extends \core\Controller
{
    private SecurityHelper $security_helper;

    public function __construct(){
        $this->response = new Response();
        $this->request = new Request();
        $this->security_helper = new SecurityHelper();
        $this->setMiddleWare(new EncryptionLayerMiddleWare());
        $this->setMiddleWare(new TrainMiddleWare(MiddleWare::MIDDLEWARE_IS_MUST));
    }

    public function index(){
        $logEventModel = new ApiLogEventModel();
        $message = "event has been logged successfully";
        $status_code = 200;
        $error = "";
        $error_code = 0;
        $logEventModel->loadData($this->request->body());
        if ($logEventModel->validate() && $logEventModel->logEvent());
        else{
            $message = "";
            $error = "failed to log event";
            $error_code = 0x3;
        }
        $response = Helper::generateApiResponse($message, $status_code, $error, $error_code);
        $this->response->returnJsonEncrypted($response, $status_code);
    }
}