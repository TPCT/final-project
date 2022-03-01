<?php

namespace core;

use core\Exceptions\RequiredFileNotFound;

class Application
{
    private static string $root_dir;
    private static string $views_dir;
    private static Application $app;

    public static Controller $controller;
    public Session $session;
    public Redirection $redirection;
    public Request $request;
    public Router $router;
    public Database $database;
    public Response $response;
    public ErrorLogger $error_logger;

    private array $config;


    public static function ROOT_DIR()
    {
        return self::$root_dir;
    }

    public static function APP()
    {
        return self::$app;
    }

    public static function VIEWS_DIR()
    {
        return self::$views_dir;
    }

    public function __construct(string $root_dir, ?string $config_file = null, string $views_dir = "views")
    {
        $this->auto_loader();
        self::$app = $this;
        self::$root_dir = $root_dir;
        self::$views_dir = rtrim($root_dir, \DIRECTORY_SEPARATOR) . \DIRECTORY_SEPARATOR . ltrim($views_dir, \DIRECTORY_SEPARATOR);
        $this->error_logger = new ErrorLogger(self::ROOT_DIR() . \DIRECTORY_SEPARATOR . "errors");
        if ($config_file)
            $this->setConfig($config_file);
        $this->router = new Router();
        $this->request = new Request();
        $this->redirection = new Redirection();
        $this->database = new Database($this->config['DB'] ?? Null);
        $this->session = new Session();
        $this->response = new Response();
    }

    private function setConfig($config_file): bool{
        if (!is_file($config_file)) {
            $this->error_logger->log(0, "couldn't find configuration file {$config_file}", __FILE__, 60);
            return False;
        }
        $environment_handler = new DotEnv($config_file);
        $environment_handler->load();
        $this->config = $_ENV;
        return True;
    }

    public function getConfig($key){
        return $this->config[$key] ?? Null;
    }

    protected function auto_loader()
    {
        spl_autoload_register([$this, 'auto_load']);
    }

    private function auto_load($class_name)
    {
        $class_path = \str_replace('\\', \DIRECTORY_SEPARATOR, $class_name) . ".php";
        $path = \rtrim(self::$root_dir, \DIRECTORY_SEPARATOR) . \DIRECTORY_SEPARATOR . $class_path;
        if (\is_file($path))
            require_once($path);
        else
            throw new RequiredFileNotFound($path);
    }

    public function run()
    {
        $response = $this->router->loadAssets();
        if ($response)
            echo $response;
        else
            echo $this->router->resolve();
    }
}
