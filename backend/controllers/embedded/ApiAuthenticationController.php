<?php

namespace controllers\embedded;

use core\Request;
use core\Response;
use helpers\Helper;
use middlewares\EncryptionLayerMiddleWare;
use models\embedded\ApiAuthenticationModel;

class ApiAuthenticationController extends \core\Controller
{
    public function __construct(){
        $this->request = new Request();
        $this->response = new Response();
        $this->setMiddleWare(new EncryptionLayerMiddleWare());
    }

    public function index(){
        $status_code = 200;
        $message = "";
        $error = "";
        $error_code = 0;

        if ($this->request->isPost()) {
            if (!Helper::isLogged()) {
                if ($this->login()) {
                    $message = "welcome {$_SESSION['user_info']['username']}.";
                } else {
                    $status_code = 403;
                    $error = "invalid authorization.";
                    $error_code = 0x11;
                }
            }else {
                $message = "success";
            }
        }else{
            $status_code = 405;
            $error = "Invalid request method";
            $error_code = 0x12;
        }

        $response = Helper::generateApiResponse($message, $status_code, $error, $error_code);
        $this->response->returnJson($response, $status_code);
    }

    public function login(){
        $api_authentication_model = new ApiAuthenticationModel();
        $api_authentication_model->loadData($this->request->body());
        if ($api_authentication_model->validate() && $api_authentication_model->login())
            return True;
        return False;
    }
}