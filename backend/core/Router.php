<?php

namespace core;

class Router
{
    protected array $routes = array();
    protected array $middlewares = array();

    public Request $request;
    public Response $response;

    public function __construct()
    {
        $this->request = new Request();
        $this->response = new Response();
    }

    public function get(string $route, $callback)
    {
        $index = strpos($route, "{");
        $parameters = [];
        
        if ($index){
            $re = '/{([a-zA-Z0-9_$]*)}/m';
            preg_match_all($re, $route, $matches, PREG_SET_ORDER, 0);
            foreach ($matches as $match){
                $parameters[] = $match[1];
            }            
            $route = substr($route, 0, $index - 1);
        }
        $this->routes['GET'][$route] = [
            'callback' => $callback,
            'parameters' => $parameters
        ];
    }

    public function post(string $route, $callback)
    {
        $index = strpos($route, "{");
        $parameters = [];
        
        if ($index){
            $re = '/@{([a-zA-Z0-9_$]*)}/m';
            preg_match_all($re, $route, $matches, PREG_SET_ORDER, 0);
            foreach ($matches as $match){
                $parameters[] = $match[1];
            }
            $route = substr($route, 0, $index - 1);
        }
        $this->routes['POST'][$route] = [
            'callback' => $callback,
            'parameters' => $parameters
        ];
    }

    public function registerMiddleWare(MiddleWare $middleware)
    {
        $this->middlewares[] = $middleware;
    }


    protected function checkMiddleWare()
    {
        foreach ($this->middlewares as $middleware) {
            if ($middleware->load())
                return True;
        }
        return False;
    }

    private function getRouteParameters($route){
        $parameters = [];
        $re = '/@{(.*)}/m';
        preg_match_all($re, $route, $matches, PREG_SET_ORDER, 0);
        foreach($matches as $match){
            $parameters[] = $match[1];
        }
        return $parameters;
    }

    public function resolve()
    {
        $route = urldecode($this->request->path());
        $method = $this->request->method();
        $routeParameters = $this->getRouteParameters($route);
        
        if ($routeParameters){
            $index = strpos($route, $routeParameters[0]) - 2;
            $route = substr($route, 0, $index);
        }

        $callback = $this->routes[$method][$route] ?? Null;

        if ($callback === Null) {
            $this->response->setStatusCode(404);
            return $this->renderView("404");
        }

        if (count($callback['parameters']) != count($routeParameters))
            $callback = Null;

        $callback = $callback['callback'];

        if (is_array($callback)) {
            [$class, $method] = $callback;
            if (\class_exists($class)) {
                $class = new $class();
                Application::$controller = $class;

                $middlewares = Application::$controller->middlewares();
                $must_block = False;
                $optional_block = [];

                foreach ($middlewares as $middleware) {
                    if ($middleware->getMiddleWareType() === MiddleWare::MIDDLEWARE_IS_MUST){
                        $must_block = !$middleware->load();
                        if ($must_block)
                            break;
                    }else{
                        $optional_block[] = !$middleware->load();
                    }
                }


                if ($must_block || (count($optional_block) && !in_array(True, $optional_block))) {
                    Application::$controller->setLayout(Null);
                    $this->response->setStatusCode(403);
                    return $this->renderView("403");
                }

                if (\method_exists($class, $method)) {
                    return \call_user_func([$class, $method], ...$routeParameters);
                }
            }
        }

        if (\is_callable($callback)) {
            return \call_user_func($callback, ...$routeParameters);
        }

        if (\is_string($callback)) {
            return $this->renderView($callback);
        }
    }

    public function mime_content_type($filename) {

        $mime_types = array(

            'txt' => 'text/plain',
            'htm' => 'text/html',
            'html' => 'text/html',
            'php' => 'text/html',
            'css' => 'text/css',
            'js' => 'application/javascript',
            'json' => 'application/json',
            'xml' => 'application/xml',
            'swf' => 'application/x-shockwave-flash',
            'flv' => 'video/x-flv',

            // images
            'png' => 'image/png',
            'jpe' => 'image/jpeg',
            'jpeg' => 'image/jpeg',
            'jpg' => 'image/jpeg',
            'gif' => 'image/gif',
            'bmp' => 'image/bmp',
            'ico' => 'image/vnd.microsoft.icon',
            'tiff' => 'image/tiff',
            'tif' => 'image/tiff',
            'svg' => 'image/svg+xml',
            'svgz' => 'image/svg+xml',

            // archives
            'zip' => 'application/zip',
            'rar' => 'application/x-rar-compressed',
            'exe' => 'application/x-msdownload',
            'msi' => 'application/x-msdownload',
            'cab' => 'application/vnd.ms-cab-compressed',

            // audio/video
            'mp3' => 'audio/mpeg',
            'qt' => 'video/quicktime',
            'mov' => 'video/quicktime',

            // adobe
            'pdf' => 'application/pdf',
            'psd' => 'image/vnd.adobe.photoshop',
            'ai' => 'application/postscript',
            'eps' => 'application/postscript',
            'ps' => 'application/postscript',

            // ms office
            'doc' => 'application/msword',
            'rtf' => 'application/rtf',
            'xls' => 'application/vnd.ms-excel',
            'ppt' => 'application/vnd.ms-powerpoint',

            // open office
            'odt' => 'application/vnd.oasis.opendocument.text',
            'ods' => 'application/vnd.oasis.opendocument.spreadsheet',
        );

        $fileInfo = explode('.',$filename);
        $ext = strtolower(array_pop($fileInfo));
        if (array_key_exists($ext, $mime_types)) {
            return $mime_types[$ext];
        }
        elseif (function_exists('finfo_open')) {
            $finfo = finfo_open(FILEINFO_MIME);
            $mimetype = finfo_file($finfo, $filename);
            finfo_close($finfo);
            return $mimetype;
        }
        else {
            return 'application/octet-stream';
        }
    }

    public function loadAssets()
    {
        $route = $this->request->path();

        if (!str_starts_with($route, "/assets"))
            return False;

        $route =  Application::APP()::VIEWS_DIR() . $route;
        if ($route) {
            $handler = file_get_contents($route);
            if ($handler) {
                $headers = [
                    'content-type: ' . $this->mime_content_type(str_replace("/", DIRECTORY_SEPARATOR, $route))
                ];
                $this->response->setStatusCode(200);
                $this->response->setResponseHeaders($headers);
                return $handler;
            }
        }
        $this->response->setStatusCode(404);
        return False;
    }

    public function renderView(string $view, $params = [])
    {
        $layoutContent = $this->renderLayoutContent();
        $viewContent = $this->renderViewContent($view, $params);
        if ($layoutContent !== Null) {
            return \str_replace('{{content}}', $viewContent, $layoutContent);
        }
        return $viewContent;
    }

    public function renderLayoutContent()
    {
        if (isset(Application::$controller) && Application::$controller->layout() !== Null) {
            \ob_start();
            include_once(Application::VIEWS_DIR() . \DIRECTORY_SEPARATOR . "layouts" . \DIRECTORY_SEPARATOR . Application::$controller->layout() . ".php");
            return \ob_get_clean();
        }
        return Null;
    }

    public function renderViewContent(string $view, $params = [])
    {
        \ob_start();
        \extract($params);
        include_once(Application::VIEWS_DIR() . \DIRECTORY_SEPARATOR . "{$view}.php");
        return \ob_get_clean();
    }
}
